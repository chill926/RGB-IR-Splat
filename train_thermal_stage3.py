"""C/R/K pilot from D: fixed epsilon0, geometry, opacity, RGB SH and thermal SH.

Uses the inherited fit-only camera signal + fixed display objective. No GT,
Oracle, K-UQ, densification or material remapping enters this first version.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
import train_thermal_joint as joint
from utils.thermal_joint import TorchFrozenDisplay, image_loss
from utils.thermal_observations import prepare_thermal_input
from utils.thermal_pseudocolor import FrozenDisplay, load_manifest, original_rgb
from utils.thermal_stage3 import (assert_frozen, atomic_save, file_sha256, lock_protocol,
                                  restore_models, validate_common, verify_inputs)
from utils.thermal_training_utils import held_lr, install_loss_scale, thermal_data_loss
from utils.thermal_spatial import TemperatureSpatialPrior
from utils.sh_utils import SH2RGB


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, pipeline = ModelParams(parser, device), PipelineParams(parser)
    parser.add_argument("--common", required=True)
    parser.add_argument("--branch", choices=("C", "R", "K"), required=True)
    parser.add_argument("--comparison_protocol", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--temperature_lr", type=float, default=None)
    parser.add_argument("--temperature_lr_final", type=float, default=None)
    parser.add_argument("--environment_lr", type=float, default=None)
    parser.add_argument("--environment_lr_final", type=float, default=None)
    parser.add_argument("--lr_hold_fraction", type=float, default=None)
    parser.add_argument("--material_lr", type=float, default=1e-4)
    parser.add_argument("--material_lr_final", type=float, default=1e-6)
    parser.add_argument("--lambda_k", type=float, default=1e-4)
    parser.add_argument("--lambda_delta_epsilon", type=float, default=1e-4)
    parser.add_argument("--lambda_spatial", type=float, default=0.)
    parser.add_argument("--spatial_neighbors", type=int, default=6)
    parser.add_argument("--spatial_temperature_scale_K", type=float, default=10.)
    parser.add_argument("--spatial_distance_cutoff", type=float, default=3.)
    parser.add_argument("--spatial_rgb_sigma", type=float, default=.15)
    parser.add_argument("--spatial_material_boundary_weight", type=float, default=.1)
    parser.add_argument("--spatial_unknown_weight", type=float, default=.1)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.save_every, args.log_every) <= 0:
        parser.error("Budget and intervals must be positive")
    if (not all(math.isfinite(v) for v in (args.material_lr, args.material_lr_final,
                                           args.lambda_k, args.lambda_delta_epsilon))
            or not 0 < args.material_lr_final <= args.material_lr
            or min(args.lambda_k, args.lambda_delta_epsilon) < 0):
        parser.error("Invalid material learning rates/regularization")
    if not math.isfinite(args.lambda_spatial) or args.lambda_spatial < 0:
        parser.error("lambda_spatial must be finite and nonnegative")
    if not args.model_path:
        parser.error("An empty, separate output directory (-m) is required")
    if args.source_path or args.load_model_path:
        parser.error("Scene/geometry paths come from the verified common checkpoint")
    return args, model, pipeline, device


def effective_training_protocol(recorded, args):
    """Override optimization settings without mutating D/common observations."""
    shared = copy.deepcopy(recorded)
    for name in ("temperature_lr", "temperature_lr_final", "environment_lr",
                 "environment_lr_final", "lr_hold_fraction"):
        value = getattr(args, name, None)
        if value is not None:
            shared[name] = value
    for source, target in (("lambda_spatial", "lambda_tv"), ("spatial_neighbors", "tv_neighbors"),
                           ("spatial_temperature_scale_K", "tv_temperature_scale_K"),
                           ("spatial_material_boundary_weight", "tv_material_boundary_weight")):
        if hasattr(args, source):
            shared[target] = getattr(args, source)
    for prefix in ("temperature", "environment"):
        start, end = (shared[prefix + suffix] for suffix in ("_lr", "_lr_final"))
        if not all(math.isfinite(value) for value in (start, end)) or not 0 < end <= start:
            raise ValueError("Invalid Stage3 " + prefix + " learning rates")
    if not math.isfinite(shared["lr_hold_fraction"]) or not 0 <= shared["lr_hold_fraction"] < 1:
        raise ValueError("Stage3 lr_hold_fraction must be finite and in [0, 1)")
    return shared


def build_context(args, model, pipeline, device, checkpoint):
    metadata = checkpoint["metadata"]
    shared = effective_training_protocol(metadata["shared_training_protocol"], args)
    args.source_path = metadata["source_path"]
    args.load_model_path = metadata["geometry_model"]
    args.eval, args.data_branch = True, "rgbt"
    args.resolution = shared["resolution"]
    dataset = model.extract(args)
    dataset.data_device, dataset.load2gpu_on_the_fly = device, True
    # Read camera/RGB and thermal canvas size; float targets come only from RJPEG.
    dataset.thermal_signal_source = True
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=metadata["geometry_iteration"], shuffle=False)
    if not scene.has_rgbt:
        raise ValueError("Stage3 pilot requires the recorded RGBT scene")
    splits = joint.select_splits(scene, metadata["camera_split"])
    for camera in scene.getTrainCameras() + scene.getTestCameras():
        joint.park_images(camera, device)
    observation = SimpleNamespace(**shared)
    observation.source_path = args.source_path
    observation.fit_camera_names = metadata["camera_split"]["fit"]
    planck = prepare_thermal_input(observation, splits["fit"] + splits["validation"],
                                  torch.device("cpu")).to(device)
    install_loss_scale(observation, splits["fit"])
    _, manifest, _ = load_manifest(args.source_path, observation.radiometric_dir,
                                   observation.radiometric_protocol)
    protocol = metadata["joint_protocol"]
    frozen = FrozenDisplay(protocol["display_calibration"], observation.radiometric_protocol,
                           observation.fit_camera_names)
    if frozen.sha256 != protocol["display_calibration_sha256"]:
        raise ValueError("Display checksum mismatch")
    display = TorchFrozenDisplay(frozen, (observation.temp_min, observation.temp_max), device)
    targets = {}
    for camera in splits["fit"] + splits["validation"]:
        array = original_rgb(args.source_path, camera.image_name, manifest["frames"][camera.image_name],
                             (camera.image_height, camera.image_width))
        targets[camera.image_name] = torch.from_numpy(array.transpose(2, 0, 1).copy())
    field, sh = restore_models(checkpoint, gaussians, args.branch, device)
    weights = SimpleNamespace(**protocol["loss_weights"])
    return SimpleNamespace(gaussians=gaussians, scene=scene, splits=splits, field=field, sh=sh,
                           planck=planck, observation=observation, display=display, targets=targets,
                           weights=weights, dataset=dataset, pipe=pipeline.extract(args),
                           background=torch.zeros(3, device=device))


@torch.no_grad()
def diagnostics(field, gradients=None):
    rows = []
    for material_id, name in enumerate(field.material_names):
        selected = field.material_ids == material_id
        values = field.temperature[selected, 0]
        epsilon = field.emissivity()[selected, 0]
        row = {"name": name, "gaussians": int(selected.sum()),
               "enabled": bool(field.learn_k_by_material[material_id]),
               "epsilon0_fixed": float(field.epsilon0_by_material[material_id]),
               "k": float(field.k_epsilon_by_material[material_id]),
               "delta_epsilon": float(field.delta_epsilon_by_material[material_id]),
               "T_mean_K": float(values.mean()) if len(values) else None,
               "T_std_K": float(values.std(unbiased=False)) if len(values) else None,
               "T_range_K": float(values.max() - values.min()) if len(values) else None,
               "epsilon_clipped_fraction": float(((epsilon <= .01) | (epsilon >= .99)).float().mean()) if len(values) else None,
               "parameter_bound_fraction": float(abs(float(field.k_epsilon_by_material[material_id])) >= .95 * field.k_max
                   if field.branch == "K" else abs(float(field.delta_epsilon_by_material[material_id])) >= .95 * field.delta_epsilon_max)}
        if gradients is not None:
            row["material_data_gradient_raw"] = gradients.get("material_data_gradient_raw", [0.] * len(field.material_names))[material_id]
        rows.append(row)
    temperature = field.temperature
    return {"environment": float(field.environment), "materials": rows,
            "temperature_bound_fraction": float(((temperature <= field.temp_min + .5) |
                                                    (temperature >= field.temp_max - .5)).float().mean()),
            "unknown_gaussians": int((field.material_ids == -1).sum())}


def run(args, model, pipeline, device):
    if device.type != "cuda":
        raise RuntimeError("Stage3 needs the server CUDA rasterizer")
    output = Path(args.model_path).resolve()
    checkpoint = torch.load(args.common, map_location="cpu")
    common = validate_common(checkpoint)
    verify_inputs(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Stage3 output is not empty; refusing overwrite/resume")
    for protected in (Path(checkpoint["metadata"]["source_path"]).resolve(),
                      Path(checkpoint["metadata"]["geometry_model"]).resolve(),
                      Path(args.common).resolve().parent,
                      Path(common["source_checkpoint"]).resolve().parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Stage3 output overlaps existing data/common/models")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    context = build_context(args, model, pipeline, device, checkpoint)
    c, field, obs = context, context.field, context.observation
    spatial_prior = TemperatureSpatialPrior.build(c.gaussians.get_xyz, c.gaussians.get_opacity,
        SH2RGB(c.gaussians._features_dc[:, 0, :]), field.material_ids, c.sh.fit_support,
        neighbors=args.spatial_neighbors, temperature_scale_K=args.spatial_temperature_scale_K,
        distance_cutoff=args.spatial_distance_cutoff, rgb_sigma=args.spatial_rgb_sigma,
        material_boundary_weight=args.spatial_material_boundary_weight, unknown_weight=args.spatial_unknown_weight)
    if args.lambda_spatial > 0 and not spatial_prior.source.numel():
        raise ValueError("No supported local edges for the spatial temperature prior")
    if args.branch != "C" and not bool(field.learn_k_by_material.any()):
        raise ValueError("No material enabled for R/K")
    protocol = {"kind": "frozen_D_C_R_K_pilot_v1", "common_sha256": file_sha256(args.common),
                "seed": args.seed, "steps": args.steps, "eval_every": args.eval_every,
                "save_every": args.save_every, "log_every": args.log_every,
                "material_lr": args.material_lr, "material_lr_final": args.material_lr_final,
                "lambda_k": args.lambda_k, "lambda_delta_epsilon": args.lambda_delta_epsilon,
                "lambda_spatial": args.lambda_spatial, "spatial_prior": spatial_prior.protocol,
                "signal_handoff": {"checkpoint": "float_T_E_material_and_frozen_SH_state",
                    "render_domain": "normalized_camera_Q", "raw_definition": "Q=DN+PlanckO",
                    "pseudo_color_inverse_used": False, "signal_channels": 1,
                    "rasterizer_channels": "same_scalar_repeated_3", "pseudo_color_role": "display_loss_and_evaluation_only"},
                "shared_training_protocol": effective_training_protocol(checkpoint["metadata"]["shared_training_protocol"], args),
                "loss_weights": {"lambda_signal": c.weights.lambda_signal, "lambda_display": c.weights.lambda_display},
                "enabled_materials": common["enabled_materials"], "epsilon0_fixed": True,
                "geometry_opacity_RGB_SH_thermal_SH_frozen": True, "K_UQ_enabled": False,
                "truth_used_for_training": False, "test_used_for_training_or_selection": False,
                "regularization_status": "explicit_pilot_values_not_synthetic_GT_tuned",
                "environment_prior_anchor": "common_environment", "initial_step_eligible_for_selection": True,
                "renderer_settings": {"sh_degree": c.dataset.sh_degree, "is_6dof": c.dataset.is_6dof,
                                      "pipeline": vars(c.pipe), "background": "black"},
                "runtime_source_sha256": {str(path.relative_to(Path(__file__).parent)): file_sha256(path)
                    for path in [Path(__file__), Path(__file__).parent / "utils/thermal_stage3.py",
                                 Path(__file__).parent / "utils/thermal_physics.py",
                                 Path(__file__).parent / "utils/thermal_joint.py",
                                 Path(__file__).parent / "utils/thermal_sh_residual.py",
                                 Path(__file__).parent / "utils/thermal_observations.py",
                                 Path(__file__).parent / "utils/thermal_training_utils.py",
                                 Path(__file__).parent / "utils/thermal_spatial.py",
                                 Path(__file__).parent / "scene/dataset_readers.py",
                                 Path(__file__).parent / "scene/__init__.py",
                                 Path(__file__).parent / "gaussian_renderer/__init__.py"]}}
    lock_sha = lock_protocol(args.comparison_protocol, protocol)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_plan.json").write_text(json.dumps(dict(protocol, branch=args.branch), indent=2) + "\n")
    (output / "spatial_prior.json").write_text(json.dumps(spatial_prior.protocol, indent=2) + "\n")
    groups = [{"name": "temperature", "params": [field.temperature_raw],
               "start": obs.temperature_lr, "end": obs.temperature_lr_final},
              {"name": "environment", "params": [field.environment_raw],
               "start": obs.environment_lr, "end": obs.environment_lr_final}]
    if args.branch != "C":
        parameter = field.k_raw if args.branch == "K" else field.delta_epsilon_raw
        groups.append({"name": "material", "params": [parameter],
                       "start": args.material_lr, "end": args.material_lr_final})
    expected = {id(p) for p in field.parameters() if p.requires_grad}
    actual = {id(p) for group in groups for p in group["params"]}
    if expected != actual or field.epsilon0_by_material.requires_grad:
        raise RuntimeError("Optimizer parameter set violates Stage3 contract")
    optimizer = torch.optim.Adam([dict(group, lr=group["start"]) for group in groups])
    anchor = field.environment.detach().clone()
    initial_T = field.temperature_raw.detach().clone()
    initial_E = field.environment_raw.detach().clone()
    def log(row):
        with (output / "stage3_training.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print("[stage3 " + args.branch + "] " + json.dumps(row, allow_nan=False), flush=True)
    def validate():
        return joint.validation(field, c.planck, c.gaussians, c.pipe, c.background, c.dataset, c.sh,
                                c.splits["validation"], c.display, c.targets, obs, c.weights)
    def save(name, step, metrics):
        assert_frozen(checkpoint, field, c.gaussians, c.sh)
        payload = field.export_state()
        payload["state_dict"] = {key: value.detach().cpu().clone() for key, value in payload["state_dict"].items()}
        metadata = copy.deepcopy(checkpoint["metadata"])
        metadata.update(stage3_common_endpoint=False, training_stage="stage3_" + args.branch,
                        shared_training_protocol=protocol["shared_training_protocol"],
                        signal_handoff_protocol=protocol["signal_handoff"],
                        stage3_protocol=protocol, stage3_common_checkpoint=str(Path(args.common).resolve()),
                        comparison_protocol=str(Path(args.comparison_protocol).resolve()),
                        comparison_protocol_sha256=lock_sha, seed=args.seed, validation_metrics=metrics)
        payload.update(step=step, metadata=metadata, joint_geometry=checkpoint["joint_geometry"],
                       sh_residual=checkpoint["sh_residual"])
        atomic_save(payload, output / name)
    metrics = validate()
    best, best_step = metrics["selection_loss"], 0
    save("thermal_" + args.branch + "_best.pt", 0, metrics)
    log({"step": 0, "validation": metrics, "diagnostics": diagnostics(field),
         "optimizer_groups": [group["name"] for group in groups]})
    order = list(range(len(c.splits["fit"])))
    generator = random.Random(args.seed)
    material_gradient_max = [0.] * len(field.material_names)
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = held_lr(step, args.steps, group["start"], group["end"], obs.lr_hold_fraction)
        if (step - 1) % len(order) == 0:
            generator.shuffle(order)
        camera = c.splits["fit"][order[(step - 1) % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        prediction = joint.render_signal(camera, field, c.planck, c.gaussians, c.pipe, c.background, c.dataset, c.sh)
        mask = camera.thermal_valid_mask.to(device)
        signal = thermal_data_loss(prediction, camera.original_physical_image[:1].to(device), obs, mask)
        display_term = image_loss(c.display(prediction, camera.image_name), c.targets[camera.image_name].to(device), mask)
        data_loss = c.weights.lambda_signal * signal + c.weights.lambda_display * display_term
        data_loss.backward()
        gradients = {"temperature_data_gradient_abs_mean": float(field.temperature_raw.grad.abs().mean()),
                     "environment_data_gradient_raw": float(field.environment_raw.grad)}
        if args.branch != "C":
            raw = field.k_raw if args.branch == "K" else field.delta_epsilon_raw
            gradients["material_data_gradient_raw"] = raw.grad.detach().cpu().tolist()
            material_gradient_max = [max(a, abs(b)) for a, b in zip(material_gradient_max, gradients["material_data_gradient_raw"])]
        environment = F.smooth_l1_loss(field.environment, anchor, beta=obs.environment_prior_beta)
        material = field.branch_regularizer(delta=0.0)
        material_weight = args.lambda_k if args.branch == "K" else args.lambda_delta_epsilon
        if args.lambda_spatial > 0:
            spatial = spatial_prior(field.temperature)
        else:
            with torch.no_grad():
                spatial = spatial_prior(field.temperature)
        regularization = obs.lambda_environment * environment + material_weight * material + args.lambda_spatial * spatial
        spatial_gradient_mean = None
        if args.lambda_spatial > 0 and (step == 1 or step % args.log_every == 0 or step == args.steps):
            spatial_gradient_mean = float(torch.autograd.grad(args.lambda_spatial * spatial,
                field.temperature_raw, retain_graph=True)[0].abs().mean())
        regularization.backward()
        if not bool(torch.isfinite(data_loss)) or not bool(torch.isfinite(regularization)):
            raise FloatingPointError("Non-finite Stage3 loss")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Missing/non-finite Stage3 gradient: " + group["name"])
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            log(dict(step=step, fit_camera=camera.image_name, signal_loss=float(signal.detach()),
                     display_loss=float(display_term.detach()), regularization_loss=float(regularization.detach()),
                     spatial_temperature_loss=float(spatial.detach()),
                     weighted_spatial_temperature_loss=float((args.lambda_spatial * spatial).detach()),
                     temperature_spatial_gradient_abs_mean=spatial_gradient_mean,
                     learning_rates={group["name"]: group["lr"] for group in optimizer.param_groups},
                     **gradients, diagnostics=diagnostics(field, gradients)))
        if step % args.eval_every == 0 or step == args.steps:
            metrics = validate()
            log({"step": step, "validation": metrics})
            if metrics["selection_loss"] < best:
                best, best_step = metrics["selection_loss"], step
                save("thermal_" + args.branch + "_best.pt", step, metrics)
        if step % args.save_every == 0 and step != args.steps:
            save("thermal_" + args.branch + "_latest.pt", step, metrics)
    save("thermal_" + args.branch + "_final.pt", args.steps, metrics)
    summary = {"branch": args.branch, "best_step": best_step, "stopped_step": args.steps,
               "best_validation_objective": best, "final_validation_metrics": metrics,
               "temperature_raw_max_update": float((field.temperature_raw - initial_T).abs().max()),
               "environment_raw_update": float((field.environment_raw - initial_E).abs().max()),
               "material_data_gradient_abs_max_by_material": material_gradient_max,
               "frozen_state_verified": True, "diagnostics": diagnostics(field), "protocol": protocol}
    with torch.no_grad():
        summary["final_spatial_temperature_loss"] = float(spatial_prior(field.temperature))
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    log({"completed": True, "best_step": best_step, "output": str(output)})


if __name__ == "__main__":
    run(*parse_args())
