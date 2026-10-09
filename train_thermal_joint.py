"""Experiment D: fresh thermal field, RGB-initialized fixed-count joint fitting.

Reference Stage2 weights are never loaded. Only config, splits and provenance
are reused. RGB and thermal graphs are backpropagated sequentially.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

import train_thermal_physics as base
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.thermal_joint import (TorchFrozenDisplay, geometry_digest, image_loss,
    joint_signal, sequential_joint_step)
from utils.thermal_physics import frozen_geometry_state
from utils.thermal_pseudocolor import FrozenDisplay, load_manifest, original_rgb
from utils.thermal_sh_residual import ThermalSHResidual
from utils.thermal_training_utils import held_lr, install_loss_scale, masked_mean, thermal_data_loss


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--joint_reference", required=True)
    parser.add_argument("--display_calibration", required=True)
    parser.add_argument("--joint_start_step", type=int, default=3000)
    parser.add_argument("--lambda_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_signal", type=float, default=1.0)
    parser.add_argument("--lambda_display", type=float, default=1.0)
    parser.add_argument("--joint_position_lr", type=float, default=1.6e-5)
    parser.add_argument("--joint_position_lr_final", type=float, default=1.6e-7)
    parser.add_argument("--joint_scaling_lr", type=float, default=5e-4)
    parser.add_argument("--joint_rotation_lr", type=float, default=1e-4)
    parser.add_argument("--joint_opacity_lr", type=float, default=5e-3)
    parser.add_argument("--joint_rgb_lr", type=float, default=2.5e-4)
    options, remaining = parser.parse_known_args()
    previous = sys.argv
    try:
        sys.argv = [previous[0]] + remaining
        args, dataset, pipe, device = base.parse_args()
    finally:
        sys.argv = previous
    if (args.stage != "stage2" or args.stage2_checkpoint or args.sh_residual_degree != 2
            or args.observation_domain != "raw_rjpeg" or args.oracle_temperature_supervision
            or args.lambda_temperature_supervision or args.lambda_tv != 0):
        raise ValueError("D requires fresh raw_rjpeg Stage2, SH2, fixed emissivity, no oracle and lambda_tv=0")
    if args.stage2_stop_mode != "budget" or not 1 <= options.joint_start_step < args.steps:
        raise ValueError("D needs a fixed budget extending beyond joint_start_step")
    if not 1 <= args.sh_residual_start_step <= args.steps:
        raise ValueError("Invalid SH start step")
    if min(args.save_every, args.eval_every) <= 0:
        raise ValueError("Save/evaluation intervals must be positive")
    for name, value in vars(options).items():
        if name.startswith("lambda_") or name.startswith("joint_") and name.endswith(("lr", "lr_final")):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite D parameter required: " + name)
    if not 0 < options.joint_position_lr_final <= options.joint_position_lr:
        raise ValueError("Invalid joint position LR schedule")
    if (not all(math.isfinite(value) for value in (args.sh_residual_bound, args.sh_residual_lr,
            args.sh_residual_lr_final, args.lambda_sh_residual, args.lambda_environment,
            args.temperature_lr, args.temperature_lr_final, args.environment_lr, args.environment_lr_final,
            args.huber_delta, args.environment_prior_beta)) or
            args.sh_residual_bound <= 0 or args.lambda_sh_residual < 0 or args.lambda_environment < 0 or
            min(args.temperature_lr, args.temperature_lr_final, args.environment_lr, args.environment_lr_final,
                args.huber_delta, args.environment_prior_beta) <= 0 or
            not 0 < args.sh_residual_lr_final <= args.sh_residual_lr or
            not args.temp_min < args.temp_ref < args.temp_max or not 0 <= args.lr_hold_fraction < 1):
        raise ValueError("Invalid inherited thermal/SH configuration")
    return args, dataset, pipe, device, options


def select_splits(scene, recorded):
    selected = {}
    for split in ("fit", "validation", "test"):
        available = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
        by_name = {camera.image_name: camera for camera in available}
        names = recorded.get(split, [])
        if not names or len(set(names)) != len(names) or len(by_name) != len(available):
            raise ValueError("Missing/duplicate cameras in " + split)
        if any(name not in by_name for name in names):
            raise ValueError("Recorded cameras missing in " + split)
        selected[split] = [by_name[name] for name in names]
    fit, validation, test = (set(recorded[split]) for split in ("fit", "validation", "test"))
    if fit & validation or (fit | validation) & test:
        raise ValueError("Overlapping D camera splits")
    if fit | validation != {camera.image_name for camera in scene.getTrainCameras()} or test != {camera.image_name for camera in scene.getTestCameras()}:
        raise ValueError("Recorded split does not cover the published split")
    return selected


def park_images(camera, device):
    # Camera transforms stay on CUDA, while all observation images live on CPU.
    for name in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"):
        setattr(camera, name, getattr(camera, name).to(device))
    for name in ("original_image", "original_rgb_image", "original_physical_image", "thermal_valid_mask"):
        value = getattr(camera, name, None)
        if value is not None:
            setattr(camera, name, value.cpu())


def render_rgb(camera, gaussians, pipe, background, dataset):
    return render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, background.device,
        dataset.is_6dof, detach_geometry=False, feature_set="rgb")["render"][:3]


def render_signal(camera, field, planck, gaussians, pipe, background, dataset, sh):
    signal = joint_signal(field, planck, gaussians, camera, sh)
    return render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0, background.device,
        dataset.is_6dof, override_color=signal.repeat(1, 3), detach_geometry=False)["render"][:1]


@torch.no_grad()
def validation(field, planck, gaussians, pipe, background, dataset, sh, cameras, display, targets, args, options):
    rows = []
    for camera in cameras:
        predicted = render_signal(camera, field, planck, gaussians, pipe, background, dataset, sh)
        target = camera.original_physical_image[:1].to(background.device)
        mask = camera.thermal_valid_mask.to(background.device)
        signal_loss = thermal_data_loss(predicted, target, args, mask)
        colored = display(predicted, camera.image_name)
        gt = targets[camera.image_name].to(background.device)
        display_loss = image_loss(colored, gt, mask)
        full_mse = float((colored - gt).square().mean())
        rgb = render_rgb(camera, gaussians, pipe, background, dataset)
        rgb_target = camera.original_rgb_image.to(background.device)
        rgb_mse = float((rgb.clamp(0, 1) - rgb_target).square().mean())
        row = {"signal_loss": float(signal_loss), "display_loss": float(display_loss),
            "selection_loss": float(options.lambda_signal * signal_loss + options.lambda_display * display_loss),
            "PSNR": -10 * math.log10(max(full_mse, 1e-12)), "RGB_PSNR": -10 * math.log10(max(rgb_mse, 1e-12)),
            "signal_MSE": float(masked_mean((predicted - target).square(), mask))}
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("Non-finite D validation")
        rows.append(row)
    result = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    result["signal_RMSE"] = math.sqrt(result["signal_MSE"])
    return result


def run(args, dataset, pipe, device, options):
    if device.type != "cuda":
        raise RuntimeError("D training requires the server CUDA rasterizer")
    reference = torch.load(options.joint_reference, map_location="cpu")
    metadata = reference["metadata"]
    if reference.get("branch") != "stage2" or reference.get("joint_geometry") is not None:
        raise ValueError("D reference must be the frozen-geometry raw_rjpeg Stage2 experiment")
    shared = metadata["shared_training_protocol"]
    if shared.get("observation_domain") != "raw_rjpeg":
        raise ValueError("D reference is not raw_rjpeg")
    output = Path(args.model_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("D output must be empty")
    for protected in (Path(args.source_path), Path(args.geometry_model), Path(options.joint_reference).resolve().parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("D output overlaps existing data/models")
    args.geometry_sha256 = base.geometry_checkpoint_sha256(args.geometry_model, args.geometry_iteration)
    if args.geometry_sha256 != metadata["geometry_sha256"]:
        raise ValueError("D Stage1 initializer changed")
    if base.file_sha256(args.material_config) != metadata["material_config_sha256"]:
        raise ValueError("D material config changed")
    for key in ("temp_min", "temp_max", "temp_ref"):
        if getattr(args, key) != reference["config"][key]:
            raise ValueError("D temperature bounds/reference differ")
    recorded = metadata["camera_split"]
    args.fit_camera_names, args.validation_camera_names, args.test_camera_names = (recorded[key] for key in ("fit", "validation", "test"))
    args.radiometric_protocol = shared["radiometric_protocol"]
    if args.radiometric_protocol.get("format_version") != 2:
        raise ValueError("D must reuse the coordinate-repaired inputs")
    directory, manifest, _ = load_manifest(args.source_path, args.radiometric_dir, args.radiometric_protocol)
    frozen_display = FrozenDisplay(options.display_calibration, args.radiometric_protocol, args.fit_camera_names)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset.model_path, dataset.load_model_path = args.model_path, args.geometry_model
    dataset.load2gpu_on_the_fly = True
    # Geometry must load onto CUDA; camera images are parked individually below.
    dataset.data_device = device
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=args.geometry_iteration, shuffle=False)
    if not scene.has_rgbt or scene.loaded_iter != args.geometry_iteration:
        raise ValueError("D requires the recorded RGBT Stage1 model")
    splits = select_splits(scene, recorded)
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    for camera in all_cameras:
        park_images(camera, device)
    # Training never decodes held-out test signals or fits their statistics.
    planck = base.prepare_thermal_input(args, splits["fit"] + splits["validation"], torch.device("cpu")).to(device)
    args.loss_scale_protocol = shared.get("loss_scale_protocol")
    install_loss_scale(args, splits["fit"])
    if args.loss_scale_protocol != shared.get("loss_scale_protocol"):
        raise ValueError("D loss scale differs from its matched reference")
    display = TorchFrozenDisplay(frozen_display, (args.temp_min, args.temp_max), device)
    targets = {}
    for camera in splits["fit"] + splits["validation"]:
        value = original_rgb(args.source_path, camera.image_name, manifest["frames"][camera.image_name],
                             (camera.image_height, camera.image_width))
        targets[camera.image_name] = torch.from_numpy(value.transpose(2, 0, 1).copy())
        if camera.original_rgb_image is None:
            raise ValueError("Missing paired RGB target")
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest",
                 "_thermal_features_dc", "_thermal_features_rest"):
        getattr(gaussians, name).requires_grad_(False)
    count = len(gaussians.get_xyz)
    background = torch.zeros(3, device=device)
    support_weights = torch.zeros(count, device=device)
    visible_color = torch.ones(count, 3, device=device)
    @torch.no_grad()
    def visibility_provider(camera):
        package = render(camera, gaussians, pipe, background, 0.0, 0.0, 0.0,
            device, dataset.is_6dof, override_color=visible_color, detach_geometry=True)
        support_weights.add_(package["visibility_filter"].to(support_weights) *
            gaussians.get_opacity.detach().reshape(-1) * package["radii"].float().square())
        return package["visibility_filter"], package["radii"]
    material_config = base.load_material_config(args.material_config)
    if abs(material_config["temperature_reference_K"] - args.temp_ref) > 1e-6:
        raise ValueError("Material reference temperature mismatch")
    args.material_config_document = material_config["document"]
    args.material_config_sha256 = base.file_sha256(args.material_config)
    field = base.make_field(args, scene, device, planck, visibility_provider, splits["fit"], material_config)
    field.set_branch_trainability()
    support = support_weights > 0
    if not bool(support.any()):
        raise ValueError("No fit support for D")
    sh = ThermalSHResidual(gaussians.get_xyz, support,
        args.sh_residual_bound * float(args.loss_scale_protocol["scale"]), trainable=False).to(device)
    del visible_color, support_weights, reference
    environment_anchor = field.environment.detach().clone()
    extent = float(scene.cameras_extent)
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid scene extent")
    groups = [
        {"name": "temperature", "params": [field.temperature_raw], "start": args.temperature_lr, "end": args.temperature_lr_final, "begin": 1},
        {"name": "environment", "params": [field.environment_raw], "start": args.environment_lr, "end": args.environment_lr_final, "begin": 1},
        {"name": "sh_residual", "params": [sh.raw], "start": args.sh_residual_lr, "end": args.sh_residual_lr_final, "begin": args.sh_residual_start_step}]
    geometry_settings = (("xyz", "_xyz", options.joint_position_lr * extent, options.joint_position_lr_final * extent),
        ("scaling", "_scaling", options.joint_scaling_lr, options.joint_scaling_lr * .01),
        ("rotation", "_rotation", options.joint_rotation_lr, options.joint_rotation_lr * .01),
        ("opacity", "_opacity", options.joint_opacity_lr, options.joint_opacity_lr * .01),
        ("rgb_dc", "_features_dc", options.joint_rgb_lr, options.joint_rgb_lr * .01),
        ("rgb_rest", "_features_rest", options.joint_rgb_lr / 20, options.joint_rgb_lr / 2000))
    for name, attribute, start, end in geometry_settings:
        groups.append({"name": name, "params": [getattr(gaussians, attribute)], "start": start, "end": end, "begin": options.joint_start_step})
    optimizer = torch.optim.Adam([dict(group, lr=0.0) for group in groups])
    output.mkdir(parents=True, exist_ok=True)
    args.warm_start_checkpoint_sha256 = None
    args.sh_residual_protocol = {"enabled": True, "degree": 2, "channels": 1, "coefficients_per_gaussian": 8,
        "DC_trainable": False, "signal_bound": sh.signal_bound, "start_step": args.sh_residual_start_step,
        "direction_geometry_gradient": True, "support": "initial_fit_frustum_nonzero_opacity"}
    protocol = {"kind": "fixed_count_rgb_ir_D_v1", "gaussian_count": count, "densification": False,
        "joint_start_step": options.joint_start_step, "reference_settings_only": str(Path(options.joint_reference).resolve()),
        "reference_sha256": base.file_sha256(options.joint_reference), "Stage2_weights_loaded": False,
        "material_assignment": "fixed_Gaussian_identity_from_initial_fit_masks", "emissivity": "fixed_material_prior",
        "display_calibration_sha256": frozen_display.sha256, "display_calibration": str(Path(options.display_calibration).resolve()),
        "display_loss": "0.8_valid_pixel_L1+0.2_valid_window_DSSIM", "rgb_loss": "0.8_L1+0.2_DSSIM",
        "signal_loss": "fit_contrast_scaled_SmoothL1", "loss_weights": {key: getattr(options, key) for key in ("lambda_rgb", "lambda_signal", "lambda_display")},
        "geometry_learning_rates": {name: [start, end] for name, _, start, end in geometry_settings},
        "selection": "minimum_validation_signal_plus_display_loss_after_joint_start",
        "test_used_for_training_or_selection": False, "sequential_modality_backward": True,
        "camera_images_on_CPU": True, "RGB_Stage1_may_have_seen_internal_validation_RGB": True}
    def save(path, step, metrics):
        state = frozen_geometry_state(gaussians)
        joint = dict(protocol, trained_geometry_sha256=geometry_digest(state))
        info = base.checkpoint_metadata(args, scene, training_stage="joint_D", joint_protocol=joint,
            validation_metrics=metrics, stage2_common_endpoint=False)
        payload = field.export_state()
        payload["state_dict"] = {key: value.detach().cpu().clone() for key, value in payload["state_dict"].items()}
        payload.update(step=int(step), metadata=info, joint_geometry=state, sh_residual=sh.export_state())
        temporary = str(path) + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, str(path))
    metrics = validation(field, planck, gaussians, pipe, background, dataset, sh, splits["validation"], display, targets, args, options)
    log_path = output / "joint_training.jsonl"
    def log(row):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print("[joint D] " + json.dumps(row, allow_nan=False), flush=True)
    log({"step": 0, "validation": metrics, "protocol": protocol})
    best, best_step = float("inf"), None
    order = list(range(len(splits["fit"])))
    generator = random.Random(args.seed)
    last_components = {}
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        joint_active = step >= options.joint_start_step
        for _, attribute, _, _ in geometry_settings:
            getattr(gaussians, attribute).requires_grad_(joint_active)
        sh.raw.requires_grad_(step >= args.sh_residual_start_step)
        for group in optimizer.param_groups:
            group["lr"] = 0.0 if step < group["begin"] else held_lr(step - group["begin"] + 1,
                args.steps - group["begin"] + 1, group["start"], group["end"], args.lr_hold_fraction)
        if (step - 1) % len(order) == 0:
            generator.shuffle(order)
        camera = splits["fit"][order[(step - 1) % len(order)]]
        def rgb_loss():
            if not joint_active:
                return background.new_zeros(())
            prediction = render_rgb(camera, gaussians, pipe, background, dataset)
            return options.lambda_rgb * image_loss(prediction, camera.original_rgb_image.to(device))
        def thermal_loss():
            prediction = render_signal(camera, field, planck, gaussians, pipe, background, dataset, sh)
            target = camera.original_physical_image[:1].to(device)
            mask = camera.thermal_valid_mask.to(device)
            signal = thermal_data_loss(prediction, target, args, mask)
            colored = display(prediction, camera.image_name)
            display_term = image_loss(colored, targets[camera.image_name].to(device), mask)
            environment = F.smooth_l1_loss(field.environment, environment_anchor, beta=args.environment_prior_beta)
            sh_prior = sh.regularizer()
            loss = options.lambda_signal * signal + options.lambda_display * display_term + args.lambda_environment * environment + args.lambda_sh_residual * sh_prior
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite D thermal loss")
            last_components.update(signal_loss=float(signal.detach()), display_loss=float(display_term.detach()),
                weighted_environment_loss=float((args.lambda_environment * environment).detach()),
                weighted_sh_prior=float((args.lambda_sh_residual * sh_prior).detach()))
            return loss
        rgb_value, thermal_value = sequential_joint_step(optimizer, rgb_loss, thermal_loss)
        if len(gaussians.get_xyz) != count:
            raise RuntimeError("D unexpectedly changed Gaussian count")
        if step == 1 or step % 100 == 0:
            log(dict(last_components, step=step, joint_active=joint_active, RGB_weighted_loss=rgb_value,
                thermal_weighted_loss=thermal_value, peak_allocated_GB=torch.cuda.max_memory_allocated(device) / 1e9,
                peak_reserved_GB=torch.cuda.max_memory_reserved(device) / 1e9))
        if step == options.joint_start_step - 1:
            save(output / "thermal_joint_D_warmup.pt", step, None)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = validation(field, planck, gaussians, pipe, background, dataset, sh, splits["validation"], display, targets, args, options)
            log({"step": step, "validation": metrics, "sh": sh.diagnostics()})
            if joint_active and metrics["selection_loss"] < best:
                best, best_step = metrics["selection_loss"], step
                save(output / "thermal_joint_D_selected.pt", step, metrics)
        if step % args.save_every == 0 and step != args.steps:
            save(output / "thermal_joint_D_latest.pt", step, None)
    save(output / "thermal_joint_D_final.pt", args.steps, metrics)
    summary = {"experiment": "D", "best_step": best_step, "stopped_step": args.steps,
        "best_validation_objective": best, "selected_checkpoint": "thermal_joint_D_selected.pt",
        "final_validation_metrics": metrics, "joint_protocol": protocol,
        "peak_allocated_GB": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_GB": torch.cuda.max_memory_reserved(device) / 1e9}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    log({"completed": True, "best_step": best_step, "output": str(output)})


if __name__ == "__main__":
    run(*parse_args())
