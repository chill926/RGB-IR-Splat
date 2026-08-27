"""Train stage 2 and fair C/K/R thermal branches on frozen RGBT geometry."""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.thermal_physics import (MaterialThermalField, UniformLWIRPlanckLUT,
    build_spatial_tv_edges, frozen_geometry_state, map_metal_masks_to_gaussians,
    map_thermal_observations_to_gaussians, save_thermal_checkpoint,
    verify_frozen_geometry_state)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, pipeline = ModelParams(parser, device), PipelineParams(parser)
    parser.add_argument("--stage", choices=("stage2", "branch"), required=True)
    parser.add_argument("--branch", choices=("C", "K", "R"), default="C")
    parser.add_argument("--geometry_model", required=True)
    parser.add_argument("--geometry_iteration", type=int, default=-1)
    parser.add_argument("--stage2_checkpoint", default="")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--k_warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature_lr", type=float, default=1e-3)
    parser.add_argument("--environment_lr", type=float, default=1e-3)
    parser.add_argument("--material_lr", type=float, default=1e-4)
    parser.add_argument("--lambda_tv", type=float, default=1e-4)
    parser.add_argument("--tv_neighbors", type=int, default=6)
    parser.add_argument("--lambda_environment", type=float, default=0.01)
    parser.add_argument("--environment_prior_beta", type=float, default=0.02)
    parser.add_argument("--environment_init_quantile", type=float, default=0.1)
    parser.add_argument("--lambda_k", type=float, default=None)
    parser.add_argument("--lambda_delta_epsilon", type=float, default=None)
    parser.add_argument("--regularization_protocol", default="",
                        help="regularization_scan.json selected on synthetic validation; required for formal K/R")
    parser.add_argument("--comparison_protocol", default="",
                        help="shared C/K/R protocol lock file; required for formal branch training")
    parser.add_argument("--regularization_scan_run", action="store_true", default=False,
                        help="Only for tools/scan_material_regularization.py, never for a reported test run")
    parser.add_argument("--sigma_k_nonmetal", type=float, default=1e-4)
    parser.add_argument("--sigma_k_metal", type=float, default=1e-3)
    parser.add_argument("--k_prior_nonmetal", type=float, default=0.0)
    parser.add_argument("--k_prior_metal", type=float, default=0.0)
    parser.add_argument("--sigma_delta_nonmetal", type=float, default=0.05)
    parser.add_argument("--sigma_delta_metal", type=float, default=0.05)
    parser.add_argument("--huber_delta", type=float, default=0.02)
    parser.add_argument("--temp_min", type=float, default=250.0)
    parser.add_argument("--temp_max", type=float, default=450.0)
    parser.add_argument("--temp_ref", type=float, default=300.0)
    parser.add_argument("--epsilon_metal", type=float, default=0.3)
    parser.add_argument("--epsilon_nonmetal", type=float, default=0.9)
    parser.add_argument("--metal_mask_dir", default="")
    parser.add_argument("--metal_vote_threshold", type=float, default=0.5)
    parser.add_argument("--allow_missing_metal_masks", action="store_true", default=False)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--stage2_patience", type=int, default=10,
                        help="Stage-2 validation checks without improvement before stopping; 0 disables")
    parser.add_argument("--stage2_min_delta", type=float, default=1e-5)
    parser.add_argument("--stage2_temperature_stability_K", type=float, default=0.01)
    parser.add_argument("--stage2_environment_stability", type=float, default=1e-5)
    parser.add_argument("--stage3_material_stability", type=float, default=1e-6)
    parser.add_argument("--observation_domain", choices=("normalized_dn", "calibrated_radiance"),
                        default="normalized_dn")
    parser.add_argument("--dn0_radiance", type=float, default=0.0,
                        help="Physical band radiance corresponding to normalized DN=0")
    parser.add_argument("--dn1_radiance", type=float, default=1.0,
                        help="Physical band radiance corresponding to normalized DN=1")
    parser.add_argument("--temperature_gt", default="",
                        help="Optional per-Gaussian synthetic temperature truth (.npy or .pt, Kelvin)")
    parser.add_argument("--lambda_temperature_supervision", type=float, default=0.0)
    args = parser.parse_args()
    args.geometry_model, args.model_path = os.path.abspath(args.geometry_model), os.path.abspath(args.model_path)
    args.source_path = os.path.abspath(args.source_path)
    if args.metal_mask_dir:
        args.metal_mask_dir = os.path.abspath(args.metal_mask_dir)
    if args.temperature_gt:
        args.temperature_gt = os.path.abspath(args.temperature_gt)
    args.data_branch = "rgbt"
    return args, model.extract(args), pipeline.extract(args), device


def observation_to_model_radiance(image, args, planck):
    if args.observation_domain == "normalized_dn":
        return image.clamp(0.0, 1.0)
    physical = args.dn0_radiance + image * (args.dn1_radiance - args.dn0_radiance)
    return planck.normalize_physical_radiance(physical)


def apply_stage2_protocol(args):
    if args.stage != "branch":
        return
    if not args.stage2_checkpoint:
        raise ValueError("--stage2_checkpoint is required for branch training")
    checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    shared = metadata.get("shared_training_protocol", {})
    for name in ("temperature_lr", "environment_lr", "lambda_tv", "lambda_environment",
                 "environment_prior_beta", "environment_init_quantile", "huber_delta",
                 "temp_min", "temp_max", "temp_ref", "observation_domain",
                 "dn0_radiance", "dn1_radiance", "temperature_gt", "lambda_temperature_supervision",
                 "tv_neighbors"):
        if name in shared:
            setattr(args, name, shared[name])


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_regularization_protocol(args):
    if args.stage != "branch":
        return
    if args.regularization_scan_run:
        print("[protocol][SCAN ONLY] regularization is being selected on synthetic validation.")
        return
    if not args.regularization_protocol:
        raise ValueError("Formal C/K/R training requires --regularization_protocol from the synthetic validation scan")
    with open(args.regularization_protocol, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("selection_dataset_role") != "synthetic_validation":
        raise ValueError("Regularization protocol is not marked as selected on synthetic validation")
    args.k_warmup_steps = int(report.get("k_warmup_steps", args.k_warmup_steps))
    if args.branch in ("K", "R"):
        selected = report.get("selected", {}).get(args.branch)
        if not selected or "regularization" not in selected:
            raise ValueError(f"Regularization protocol has no selected value for branch {args.branch}")
        value = float(selected["regularization"])
        if args.branch == "K":
            args.lambda_k = value
        else:
            args.lambda_delta_epsilon = value
    args.regularization_protocol = os.path.abspath(args.regularization_protocol)
    args.regularization_protocol_sha256 = file_sha256(args.regularization_protocol)


def enforce_comparison_protocol(args):
    if args.stage != "branch" or args.regularization_scan_run:
        return
    if not args.comparison_protocol:
        raise ValueError("Formal C/K/R training requires the same --comparison_protocol path")
    path = os.path.abspath(args.comparison_protocol)
    protocol = {
        "format_version": 1,
        "stage2_checkpoint_sha256": file_sha256(args.stage2_checkpoint),
        "regularization_protocol_sha256": args.regularization_protocol_sha256,
        "geometry_sha256": args.geometry_sha256,
        "steps": args.steps,
        "seed": args.seed,
        "temperature_lr": args.temperature_lr,
        "environment_lr": args.environment_lr,
        "material_lr": args.material_lr,
        "lambda_tv": args.lambda_tv,
        "lambda_environment": args.lambda_environment,
        "environment_prior_beta": args.environment_prior_beta,
        "environment_init_quantile": args.environment_init_quantile,
        "huber_delta": args.huber_delta,
        "eval_every": args.eval_every,
        "k_warmup_steps": args.k_warmup_steps,
        "stage3_material_stability": args.stage3_material_stability,
    }
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        differences = {key: (existing.get(key), value) for key, value in protocol.items()
                       if existing.get(key) != value}
        if differences:
            raise ValueError(f"C/K/R comparison protocol mismatch: {differences}")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True)
    args.comparison_protocol = path
    args.comparison_protocol_sha256 = file_sha256(path)


def checkpoint_metadata(args, scene, **extra):
    metadata = {
        "geometry_model": args.geometry_model,
        "geometry_iteration": scene.loaded_iter,
        "geometry_sha256": args.geometry_sha256,
        "source_path": os.path.abspath(args.source_path),
        "seed": args.seed,
        "shared_training_protocol": {
            "temperature_lr": args.temperature_lr, "environment_lr": args.environment_lr,
            "lambda_tv": args.lambda_tv, "lambda_environment": args.lambda_environment,
            "environment_prior_beta": args.environment_prior_beta,
            "environment_init_quantile": args.environment_init_quantile,
            "huber_delta": args.huber_delta, "temp_min": args.temp_min, "temp_max": args.temp_max,
            "temp_ref": args.temp_ref, "observation_domain": args.observation_domain,
            "dn0_radiance": args.dn0_radiance, "dn1_radiance": args.dn1_radiance,
            "temperature_gt": args.temperature_gt,
            "lambda_temperature_supervision": args.lambda_temperature_supervision,
            "tv_neighbors": args.tv_neighbors,
        },
    }
    if getattr(args, "comparison_protocol", ""):
        metadata["comparison_protocol"] = args.comparison_protocol
        metadata["comparison_protocol_sha256"] = getattr(args, "comparison_protocol_sha256", "")
    if getattr(args, "regularization_protocol", ""):
        metadata["regularization_protocol"] = args.regularization_protocol
        metadata["regularization_protocol_sha256"] = getattr(args, "regularization_protocol_sha256", "")
    metadata.update(extra)
    return metadata


def geometry_checkpoint_sha256(model_path, iteration):
    path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_temperature_truth(path, count, device):
    if not path:
        return None
    if path.endswith(".npy"):
        value = torch.from_numpy(np.load(path))
    else:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value.get("temperature", value.get("temperature_K"))
    if value is None:
        raise ValueError("Temperature truth file must contain 'temperature' or 'temperature_K'")
    value = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1, 1)
    if value.shape[0] != count:
        raise ValueError(f"Temperature truth has {value.shape[0]} values, expected {count}")
    return value


def make_field(args, scene, device, planck, visibility_provider):
    if args.stage == "branch":
        if not args.stage2_checkpoint:
            raise ValueError("--stage2_checkpoint is required for branch training")
        checkpoint = torch.load(args.stage2_checkpoint, map_location=device)
        if checkpoint.get("branch") != "stage2":
            raise ValueError("C/K/R must start from a stage2 checkpoint")
        metadata = checkpoint.get("metadata", {})
        if not metadata.get("stage2_common_endpoint", False):
            raise ValueError("C/K/R must start from thermal_stage2_common.pt, not a best/intermediate checkpoint")
        expected_geometry = os.path.abspath(metadata.get("geometry_model", args.geometry_model))
        if expected_geometry != args.geometry_model:
            raise ValueError(f"Geometry mismatch: stage2 used {expected_geometry}, branch requested {args.geometry_model}")
        expected_iteration = metadata.get("geometry_iteration")
        if expected_iteration is not None and int(expected_iteration) != int(scene.loaded_iter):
            raise ValueError(f"Geometry iteration mismatch: stage2 used {expected_iteration}, loaded {scene.loaded_iter}")
        expected_hash = metadata.get("geometry_sha256")
        if expected_hash and expected_hash != args.geometry_sha256:
            raise ValueError("Geometry checkpoint content hash differs from the stage2 common checkpoint")
        expected_source = os.path.abspath(metadata.get("source_path", args.source_path))
        if expected_source != os.path.abspath(args.source_path):
            raise ValueError(f"Dataset mismatch: stage2 used {expected_source}, branch requested {args.source_path}")
        verify_frozen_geometry_state(scene.gaussians, checkpoint.get("frozen_geometry"))
        return MaterialThermalField.from_checkpoint(checkpoint, args.branch, device)
    labels = map_metal_masks_to_gaussians(scene.gaussians, scene.getTrainCameras(),
                                           args.metal_mask_dir, args.metal_vote_threshold,
                                           args.allow_missing_metal_masks, visibility_provider)
    observed = map_thermal_observations_to_gaussians(
        scene.gaussians, scene.getTrainCameras(),
        observation_transform=lambda image: observation_to_model_radiance(image, args, planck),
        visibility_provider=visibility_provider,
    )
    epsilon0 = torch.where(labels[:, None].bool(),
                           observed.new_tensor(args.epsilon_metal),
                           observed.new_tensor(args.epsilon_nonmetal))
    initial_environment = float(torch.quantile(observed, args.environment_init_quantile))
    initial_emission = ((observed - (1.0 - epsilon0) * initial_environment) /
                        epsilon0.clamp_min(1e-4)).clamp(1e-4, 1 - 1e-4)
    initial_temperature = planck.inverse(initial_emission)
    initial_temperature_unit = (initial_temperature - args.temp_min) / (args.temp_max - args.temp_min)
    return MaterialThermalField(labels, initial_temperature_unit, branch="stage2",
        temp_min=args.temp_min, temp_max=args.temp_max, temp_ref=args.temp_ref,
        epsilon_metal=args.epsilon_metal, epsilon_nonmetal=args.epsilon_nonmetal,
        initial_environment=initial_environment).to(device)


@torch.no_grad()
def validation_radiance_loss(field, planck, cameras, gaussians, pipe, background, dataset, args):
    if not cameras:
        return None
    radiance = field.radiance(planck).repeat(1, 3)
    total = 0.0
    for camera in cameras:
        prediction = render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device=background.device,
            is_6dof=dataset.is_6dof, override_color=radiance, detach_geometry=True)["render"]
        target = camera.original_physical_image if camera.original_physical_image is not None else camera.original_image
        target = target.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        target = observation_to_model_radiance(target, args, planck)
        total += float(F.smooth_l1_loss(prediction, target, beta=args.huber_delta))
    return total / len(cameras)


def main():
    args, dataset, pipe, device = parse_args()
    apply_stage2_protocol(args)
    apply_regularization_protocol(args)
    if args.steps <= 0 or args.save_every <= 0 or args.eval_every <= 0:
        raise ValueError("--steps, --save_every and --eval_every must all be positive")
    if not args.temp_min < args.temp_ref < args.temp_max:
        raise ValueError("Temperature bounds must satisfy temp_min < temp_ref < temp_max")
    if not (0.01 <= args.epsilon_metal <= 0.99 and 0.01 <= args.epsilon_nonmetal <= 0.99):
        raise ValueError("Both emissivity anchors must lie in [0.01, 0.99]")
    if args.epsilon_metal >= args.epsilon_nonmetal:
        raise ValueError("The binary metal/non-metal prior requires epsilon_metal < epsilon_nonmetal")
    if not 0.0 <= args.environment_init_quantile <= 1.0:
        raise ValueError("--environment_init_quantile must lie in [0, 1]")
    if min(args.temperature_lr, args.environment_lr, args.material_lr, args.huber_delta,
           args.environment_prior_beta) <= 0.0:
        raise ValueError("Learning rates and Huber beta values must be positive")
    if min(args.lambda_tv, args.lambda_environment, args.lambda_temperature_supervision) < 0.0:
        raise ValueError("Loss weights must be non-negative")
    if args.tv_neighbors < 1:
        raise ValueError("--tv_neighbors must be at least 1")
    if min(args.sigma_k_nonmetal, args.sigma_k_metal,
           args.sigma_delta_nonmetal, args.sigma_delta_metal) <= 0.0:
        raise ValueError("All material-prior sigma values must be positive")
    if args.stage == "branch" and args.branch == "K" and (args.lambda_k is None or args.lambda_k < 0.0):
        raise ValueError("Branch K requires a non-negative lambda_k from the scan protocol")
    if args.stage == "branch" and args.branch == "R" and (
            args.lambda_delta_epsilon is None or args.lambda_delta_epsilon < 0.0):
        raise ValueError("Branch R requires a non-negative lambda_delta_epsilon from the scan protocol")
    if args.stage == "stage2" and args.stage2_patience <= 0:
        raise ValueError(
            "Stage 2 requires --stage2_patience > 0; the common endpoint must satisfy "
            "both validation plateau and T/E stability"
        )
    if args.stage == "branch" and not 0 <= args.k_warmup_steps <= args.steps:
        raise ValueError("--k_warmup_steps must lie inside the total branch budget [0, --steps]")
    if args.stage == "branch" and args.branch == "K" and args.material_lr >= args.temperature_lr:
        raise ValueError("Branch K requires --material_lr lower than --temperature_lr")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.model_path, exist_ok=True)
    dataset.model_path, dataset.load_model_path = args.model_path, args.geometry_model
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=args.geometry_iteration, shuffle=False)
    if scene.loaded_iter is None:
        raise ValueError("Stage 2/3 requires a saved stage-1 geometry checkpoint; use --geometry_iteration -1 or an explicit iteration")
    args.geometry_sha256 = geometry_checkpoint_sha256(args.geometry_model, scene.loaded_iter)
    if not scene.has_rgbt:
        raise RuntimeError("train_thermal_physics.py currently supports RGBT-Scenes only")
    if not scene.getTrainCameras():
        raise RuntimeError("No training cameras were loaded")
    if not scene.getTestCameras():
        raise RuntimeError("A held-out validation split is required by the stopping/evaluation protocol")
    for parameter in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                      gaussians._thermal_features_dc, gaussians._thermal_features_rest,
                      gaussians._opacity, gaussians._scaling, gaussians._rotation):
        if torch.is_tensor(parameter): parameter.requires_grad_(False)

    planck = UniformLWIRPlanckLUT(args.temp_min, args.temp_max).to(device)
    if args.observation_domain == "normalized_dn":
        print("[observation] normalized_dn: temperatures are qualitative model-space estimates, not calibrated absolute truth.")
    elif args.dn1_radiance <= args.dn0_radiance:
        raise ValueError("--dn1_radiance must be greater than --dn0_radiance")
    visibility_color = torch.ones((gaussians.get_xyz.shape[0], 3), device=device)
    def visibility_provider(camera):
        package = render(camera, gaussians, pipe, torch.zeros(3, device=device), 0.0, 0.0, 0.0,
                         device, dataset.is_6dof, override_color=visibility_color,
                         detach_geometry=True)
        return package["visibility_filter"], package["radii"]
    enforce_comparison_protocol(args)
    field = make_field(args, scene, device, planck, visibility_provider)
    temperature_truth = load_temperature_truth(args.temperature_gt, gaussians.get_xyz.shape[0], device)
    pair_i, pair_j, pair_w = build_spatial_tv_edges(gaussians.get_xyz, args.tv_neighbors)
    environment_anchor = field.environment.detach().clone()
    optimizer_groups = [
        {"params": [field.temperature_raw], "lr": args.temperature_lr},
        {"params": [field.environment_raw], "lr": args.environment_lr},
    ]
    if field.branch == "K":
        optimizer_groups.append({"params": [field.k_raw], "lr": args.material_lr})
    elif field.branch == "R":
        optimizer_groups.append({"params": [field.delta_epsilon_raw], "lr": args.material_lr})
    optimizer = torch.optim.Adam(optimizer_groups)
    cameras, background = scene.getTrainCameras(), torch.zeros(3, device=device)
    camera_order = list(range(len(cameras)))
    camera_rng = random.Random(args.seed)
    validation_cameras = scene.getTestCameras()
    log_path = os.path.join(args.model_path, "thermal_training.jsonl")
    best_validation, stale_checks, final_step = float("inf"), 0, 0
    stage2_converged = False
    previous_eval_temperature = field.temperature.detach().clone()
    previous_eval_environment = field.environment.detach().clone()
    previous_eval_material = (field.k_epsilon_by_material.detach().clone()
                              if field.branch == "K" else field.delta_epsilon_by_material.detach().clone())

    for step in range(1, args.steps + 1):
        final_step = step
        warmup = args.stage == "branch" and args.branch == "K" and step <= args.k_warmup_steps
        field.set_warmup_trainability(warmup)
        camera_position = (step - 1) % len(cameras)
        if camera_position == 0:
            camera_rng.shuffle(camera_order)
        camera = cameras[camera_order[camera_position]]
        radiance = field.radiance(planck)
        prediction = render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device,
            dataset.is_6dof, override_color=radiance.repeat(1, 3), detach_geometry=True)["render"]
        target = camera.original_physical_image if camera.original_physical_image is not None else camera.original_image
        target = target.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        target = observation_to_model_radiance(target, args, planck)
        rad_loss = F.smooth_l1_loss(prediction, target, beta=args.huber_delta)
        if pair_i.numel():
            temperature = field.temperature.reshape(-1)
            tv_loss = (pair_w * (temperature[pair_i] - temperature[pair_j]).abs()).sum() / pair_w.sum()
        else:
            tv_loss = rad_loss.new_zeros(())
        environment_loss = F.smooth_l1_loss(
            field.environment, environment_anchor, beta=args.environment_prior_beta)
        material_loss = field.branch_regularizer(
            sigma_k=(args.sigma_k_nonmetal, args.sigma_k_metal),
            sigma_delta=(args.sigma_delta_nonmetal, args.sigma_delta_metal),
            k_prior=(args.k_prior_nonmetal, args.k_prior_metal),
        )
        supervised_temperature_loss = (
            F.smooth_l1_loss(field.temperature, temperature_truth)
            if temperature_truth is not None else rad_loss.new_zeros(())
        )
        loss = rad_loss + args.lambda_tv * tv_loss + args.lambda_environment * environment_loss
        branch_regularization_weight = (
            args.lambda_k if field.branch == "K"
            else (args.lambda_delta_epsilon if field.branch == "R" else 0.0)
        )
        loss = loss + branch_regularization_weight * material_loss
        loss = loss + args.lambda_temperature_supervision * supervised_temperature_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            has_temperature_truth = temperature_truth is not None
            temperature_key = ("apparent_temperature_mean_K"
                               if args.observation_domain == "normalized_dn" and not has_temperature_truth
                               else "temperature_mean_K")
            temperature_semantics = (
                "synthetic_supervised_thermodynamic" if has_temperature_truth
                else ("uncalibrated_apparent_proxy" if args.observation_domain == "normalized_dn"
                      else "calibrated_thermodynamic")
            )
            record = {"step": step, "branch": field.branch, "warmup": warmup,
                "loss": float(loss.detach()), "radiance_loss": float(rad_loss.detach()),
                "tv_loss": float(tv_loss.detach()), "environment_mean": float(field.environment.detach().mean()),
                temperature_key: float(field.temperature.detach().mean()),
                "temperature_semantics": temperature_semantics,
                "k_nonmetal": float(field.k_epsilon_by_material[0].detach()),
                "k_metal": float(field.k_epsilon_by_material[1].detach()),
                "delta_epsilon_nonmetal": float(field.delta_epsilon_by_material[0].detach()),
                "delta_epsilon_metal": float(field.delta_epsilon_by_material[1].detach())}
            if temperature_truth is not None:
                temperature_error = field.temperature.detach() - temperature_truth
                record["temperature_mae_K"] = float(temperature_error.abs().mean())
                record["temperature_rmse_K"] = float(temperature_error.square().mean().sqrt())
            print(json.dumps(record))
            with open(log_path, "a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            save_thermal_checkpoint(os.path.join(args.model_path, f"thermal_{field.branch}_step_{step}.pt"),
                field, step, checkpoint_metadata(args, scene))
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            validation = validation_radiance_loss(field, planck, validation_cameras, gaussians,
                pipe, background, dataset, args)
            if validation is not None:
                temperature_update = float((field.temperature.detach() - previous_eval_temperature).abs().mean())
                environment_update = float((field.environment.detach() - previous_eval_environment).abs().mean())
                current_material = (field.k_epsilon_by_material.detach()
                                    if field.branch == "K" else field.delta_epsilon_by_material.detach())
                material_update = float((current_material - previous_eval_material).abs().mean())
                previous_eval_temperature = field.temperature.detach().clone()
                previous_eval_environment = field.environment.detach().clone()
                previous_eval_material = current_material.clone()
                stable = (temperature_update <= args.stage2_temperature_stability_K and
                          environment_update <= args.stage2_environment_stability and
                          (field.branch in ("stage2", "C") or
                           material_update <= args.stage3_material_stability))
                update_key = ("apparent_temperature_update_K"
                              if args.observation_domain == "normalized_dn" and temperature_truth is None
                              else "temperature_update_K")
                evaluation_record = {"step": step, "branch": field.branch,
                    "validation_radiance_loss": validation, update_key: temperature_update,
                    "environment_update": environment_update,
                    "material_parameter_update": material_update,
                    "parameters_stable": stable}
                print(json.dumps(evaluation_record))
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(evaluation_record) + "\n")
                if args.stage == "stage2" and validation < best_validation - args.stage2_min_delta:
                    best_validation, stale_checks = validation, 0
                    save_thermal_checkpoint(os.path.join(args.model_path, "thermal_stage2_best.pt"), field, step,
                        checkpoint_metadata(args, scene, validation_radiance_loss=validation))
                elif args.stage == "stage2":
                    stale_checks += 1
                    if args.stage2_patience > 0 and stale_checks >= args.stage2_patience and stable:
                        stage2_converged = True
                        print(f"[stage2] validation plateau and T/E stability reached at step {step}; stopping.")
                        break

    final_metadata = checkpoint_metadata(args, scene, best_validation_radiance_loss=best_validation,
                                         stage2_common_endpoint=False)
    save_thermal_checkpoint(os.path.join(args.model_path, f"thermal_{field.branch}_final.pt"),
                            field, final_step, final_metadata)
    if args.stage == "stage2":
        if not stage2_converged:
            raise RuntimeError(
                "Stage 2 reached --steps without satisfying both validation plateau and T/E stability. "
                "Increase --steps or inspect the physical fit; thermal_stage2_common.pt was not created."
            )
        common_metadata = checkpoint_metadata(args, scene, best_validation_radiance_loss=best_validation,
                                              stage2_common_endpoint=True)
        save_thermal_checkpoint(os.path.join(args.model_path, "thermal_stage2_common.pt"),
                                field, final_step, common_metadata,
                                frozen_geometry=frozen_geometry_state(scene.gaussians))


if __name__ == "__main__":
    main()
