"""Free grayscale diagnostic on exactly the stage-2 geometry and camera split.

Place this script in graduate/. Run it in the server's physir environment.
Only one directly optimized grayscale value per Gaussian is trained. Geometry,
opacity and SH stay frozen. No temperature, emissivity, environment, TV, or
view-dependent SH is optimized. Grayscale is projected into [0, 1] after Adam.
The stage-2 checkpoint supplies initialization, split, and provenance only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.thermal_opacity import IROpacityCorrection
from scene import GaussianModel, Scene
from utils.loss_utils import ssim
from utils.thermal_training_utils import thermal_data_loss, held_lr, masked_mean
from utils.thermal_observations import prepare_thermal_input, radiometric_image_errors
from utils.thermal_physics import (
    MaterialThermalField, verify_frozen_geometry_state,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def append_log(path, value):
    line = json.dumps(value, allow_nan=False)
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def select_splits(scene, metadata):
    recorded = metadata.get("camera_split", {})
    train, test = scene.getTrainCameras(), scene.getTestCameras()
    selected = {}
    for split in ("fit", "validation", "test"):
        available = test if split == "test" else train
        names = recorded.get(split)
        by_name = {view.image_name: view for view in available}
        if not names or len(by_name) != len(available) or len(set(names)) != len(names):
            raise ValueError("Missing or ambiguous checkpoint camera split: " + split)
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError("Recorded cameras are missing: " + str(missing))
        selected[split] = [by_name[name] for name in names]
    name_sets = {key: set(recorded[key]) for key in selected}
    if name_sets["fit"] & name_sets["validation"]:
        raise ValueError("Fit and internal validation overlap")
    if (name_sets["fit"] | name_sets["validation"]) & name_sets["test"]:
        raise ValueError("Published test cameras overlap training cameras")
    if name_sets["fit"] | name_sets["validation"] != {view.image_name for view in train}:
        raise ValueError("Fit + internal validation must cover the published training split")
    if name_sets["test"] != {view.image_name for view in test}:
        raise ValueError("Recorded test split differs from the published test folders")
    return selected


def target_image(camera, device):
    image = getattr(camera, "original_physical_image", None)
    if image is None:
        image = camera.original_image
    target = image.to(device).mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("Non-finite target: " + camera.image_name)
    return target


def prediction_image(camera, gray, gaussians, pipe, background, dataset, device, override_opacity=None):
    prediction = render(
        camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device,
        dataset.is_6dof, override_color=gray.repeat(1, 3),
        detach_geometry=True, override_opacity=override_opacity)["render"][:1]
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Non-finite prediction: " + camera.image_name)
    return prediction


@torch.no_grad()
def evaluate(views, gray, gaussians, pipe, background, dataset, device, beta,
             directory=None, save_images=False, radiometric_response=None, loss_config=None, override_opacity=None):
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        if save_images:
            for folder in ("renders", "gt", "comparisons", "float_arrays"):
                (directory / folder).mkdir(exist_ok=True)
    rows = []
    for camera in tqdm(views, desc="Evaluate", leave=False):
        prediction = prediction_image(camera, gray, gaussians, pipe, background, dataset, device, override_opacity)
        target = target_image(camera, device)
        if prediction.shape != target.shape:
            raise ValueError("Image dimensions differ: " + camera.image_name)
        error = prediction - target
        mask = getattr(camera, "thermal_valid_mask", None)
        mse = float(masked_mean(error.square(), mask))
        rows.append({
            "image_name": camera.image_name,
            "PSNR": -10.0 * math.log10(max(mse, 1e-12)),
            "SSIM": float(ssim(prediction[None], target[None])),
            "MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(masked_mean(error.abs(), mask)),
            "radiance_loss": float(thermal_data_loss(prediction, target, loss_config or {"huber_delta": beta}, mask)),
            "normalized_signal_huber_loss": float(masked_mean(F.smooth_l1_loss(prediction, target, beta=beta, reduction="none"), mask)),
        })
        if radiometric_response is not None:
            rows[-1].update(radiometric_image_errors(prediction, target, radiometric_response, mask))
        if directory is not None and save_images:
            np.save(directory / "float_arrays" / (camera.image_name + ".prediction.npy"), prediction.cpu().numpy())
            np.save(directory / "float_arrays" / (camera.image_name + ".gt.npy"), target.cpu().numpy())
            filename = camera.image_name + ".png"
            save_image(prediction.clamp(0, 1), directory / "renders" / filename)
            save_image(target, directory / "gt" / filename)
            save_image(torch.cat((target, prediction, error.abs()), dim=-1).clamp(0, 1),
                       directory / "comparisons" / filename)
    keys = ("PSNR", "SSIM", "MSE", "MAE", "radiance_loss", "normalized_signal_huber_loss")
    if radiometric_response is not None:
        keys += ("camera_signal_MAE", "camera_signal_RMSE", "apparent_temperature_MAE_K")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    summary.update(RMSE=math.sqrt(summary["MSE"]), views=len(rows))
    if radiometric_response is not None:
        span = float(radiometric_response.physical_radiance_max - radiometric_response.physical_radiance_min)
        summary["camera_signal_RMSE"] = summary["RMSE"] * span
    if directory is not None:
        write_json(directory / "per_view.json", rows)
        write_json(directory / "metrics.json", summary)
    return summary


def save_control(path, gray, step, validation, protocol):
    torch.save({
        "format_version": 1, "kind": "free_gray_control", "step": int(step),
        "gray": gray.detach().cpu().clone(), "validation": validation,
        "protocol": protocol,
    }, path)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    model, pipeline = ModelParams(parser, device), PipelineParams(parser)
    parser.add_argument("--stage2_checkpoint", required=True,
                        help="Existing common checkpoint: exact split and initial radiance")
    parser.add_argument("--geometry_model", default="", help="Override if geometry directory moved")
    parser.add_argument("--radiometric_dir", default="", help="Optional moved RJPEG signal directory")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--gray_lr", type=float, default=None)
    parser.add_argument("--gray_lr_final", type=float, default=None)
    parser.add_argument("--lr_hold_fraction", type=float, default=None)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the CUDA server in the physir environment")
    if min(args.steps, args.eval_every, args.save_every) <= 0:
        raise ValueError("Step counts and evaluation/save intervals must be positive")
    if not args.model_path:
        raise ValueError("Use -m to specify a new, separate output directory")
    checkpoint_path = Path(args.stage2_checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("sh_residual") is not None:
        raise ValueError("Free scalar control cannot inherit directional SH; use a no-SH Stage2 reference")
    if checkpoint.get("branch") != "stage2":
        raise ValueError("Reference must be a stage2 checkpoint, not C/K/R")
    metadata = checkpoint.get("metadata", {})
    shared = metadata.get("shared_training_protocol", {})
    if shared.get("observation_domain") not in ("normalized_dn", "raw_rjpeg"):
        raise ValueError("This diagnostic supports normalized_dn and raw_rjpeg experiments only")
    if shared["observation_domain"] == "raw_rjpeg" and not isinstance(shared.get("radiometric_protocol"), dict):
        raise ValueError("RJPEG checkpoint is missing its radiometric calibration protocol")
    scaled = (shared.get("loss_scale_protocol") or {}).get("mode") == "fit_quantile"
    args.gray_lr = args.gray_lr if args.gray_lr is not None else (1e-4 if scaled else 1e-2)
    args.gray_lr_final = args.gray_lr_final if args.gray_lr_final is not None else (1e-6 if scaled else 1e-4)
    args.lr_hold_fraction = args.lr_hold_fraction if args.lr_hold_fraction is not None else shared.get("lr_hold_fraction", 0.0)
    if not 0 < args.gray_lr_final <= args.gray_lr or not math.isfinite(args.gray_lr) or not 0 <= args.lr_hold_fraction < 1:
        raise ValueError("Invalid gray learning rates or hold fraction")
    beta = float(shared.get("huber_delta", 0.02))
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Reference Huber delta must be finite and positive")
    source = args.source_path or metadata.get("source_path", "")
    geometry = args.geometry_model or metadata.get("geometry_model", "")
    if not source or not geometry:
        raise ValueError("Source and geometry paths are required")
    source, geometry = Path(source).resolve(), Path(geometry).resolve()
    output = Path(args.model_path).resolve()
    # Keep all outputs outside the existing data/model/checkpoint directories.
    for protected in (source, geometry, checkpoint_path.parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output must be separate from the input directory: " + str(protected))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new -m directory")
    iteration = int(metadata.get("geometry_iteration", 0))
    if iteration <= 0:
        raise ValueError("Reference must record an explicit geometry iteration")
    geometry_hash = metadata.get("geometry_sha256")
    ply = geometry / "point_cloud" / ("iteration_" + str(iteration)) / "point_cloud.ply"
    if not geometry_hash or sha256(ply) != geometry_hash:
        raise ValueError("Geometry hash differs from the stage2 reference")
    if not checkpoint.get("frozen_geometry"):
        raise ValueError("Reference must contain its frozen_geometry tensors")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.source_path, args.model_path = str(source), str(output)
    args.load_model_path, args.data_branch, args.eval = str(geometry), "rgbt", True
    # Match thermal stage2 camera loading even if this CLI flag was supplied.
    args.load2gpu_on_the_fly = False
    dataset, pipe = model.extract(args), pipeline.extract(args)
    output.mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    if not scene.has_rgbt:
        raise ValueError("This diagnostic requires an RGBT scene")
    verify_frozen_geometry_state(gaussians, checkpoint["frozen_geometry"])
    views = select_splits(scene, metadata)
    for name in ("_xyz", "_features_dc", "_features_rest", "_thermal_features_dc",
                 "_thermal_features_rest", "_opacity", "_scaling", "_rotation"):
        parameter = getattr(gaussians, name, None)
        if torch.is_tensor(parameter):
            parameter.requires_grad_(False)
    with torch.no_grad():
        field = MaterialThermalField.from_checkpoint(
            checkpoint, "stage2", device, reset_branch_parameters=False)
        config = checkpoint["config"]
        args.observation_domain = shared["observation_domain"]
        args.temp_min, args.temp_max = config["temp_min"], config["temp_max"]
        args.radiometric_protocol = shared.get("radiometric_protocol")
        args.fit_camera_names = metadata["camera_split"]["fit"]
        args.radiometric_dir = args.radiometric_dir or shared.get("radiometric_dir", "")
        planck = prepare_thermal_input(args, scene.getTrainCameras() + scene.getTestCameras(), device)
        initial_gray = field.radiance(planck).detach().clone()
        opacity_field = IROpacityCorrection.from_checkpoint(checkpoint, gaussians.get_opacity, trainable=False)
        override_opacity = opacity_field.opacity.detach() if opacity_field is not None else None
    if initial_gray.shape != (gaussians.get_xyz.shape[0], 1):
        raise ValueError("Reference field and geometry have different Gaussian counts")
    if not bool(torch.isfinite(initial_gray).all()):
        raise FloatingPointError("Reference initial radiance is non-finite")
    del field, checkpoint
    radiometric_response = planck if args.observation_domain == "raw_rjpeg" else None
    gray = torch.nn.Parameter(initial_gray.clone())
    background = torch.zeros(3, device=device)
    protocol = {
        "kind": "free_gray_control", "physical_temperature_meaning": False,
        "stage2_checkpoint": str(checkpoint_path), "stage2_sha256": sha256(checkpoint_path),
        "geometry_model": str(geometry), "geometry_iteration": iteration,
        "geometry_sha256": geometry_hash, "source_path": str(source),
        "camera_split": {key: [camera.image_name for camera in cameras]
                         for key, cameras in views.items()},
        "split_roles": {"fit": "subset_of_published_train",
                        "validation": "internal_holdout_of_published_train",
                        "test": "published_rgb_test_and_thermal_test"},
        "seed": args.seed, "steps": args.steps, "gray_lr": args.gray_lr,
        "gray_lr_final": args.gray_lr_final, "eval_every": args.eval_every,
        "resolution": args.resolution, "observation_domain": args.observation_domain,
        "loss": "scaled Huber only", "huber_delta": beta, "gray_bounds": [0, 1],
        "loss_scale_protocol": shared.get("loss_scale_protocol"), "lr_hold_fraction": args.lr_hold_fraction,
        "initialization": "per_Gaussian_stage2_reference_radiance",
        "ir_opacity_enabled": override_opacity is not None,
        "ir_opacity_trainable": False,
        "ir_opacity_source": "stage2_reference_checkpoint",
        "selection": "minimum_internal_validation_Huber_including_initialization",
        "metric_data_range": 1.0, "ssim_window_size": 11,
        "comparison_order": ["ground_truth", "prediction", "absolute_error"],
    }
    if args.observation_domain == "raw_rjpeg":
        protocol["radiometric_protocol"] = args.radiometric_protocol
    write_json(output / "protocol.json", protocol)
    print("[split] fit=%d, internal_validation=%d, published_test=%d" %
          (len(views["fit"]), len(views["validation"]), len(views["test"])), flush=True)
    log = output / "gray_training.jsonl"
    initial_metrics = {}
    for split in ("fit", "validation"):
        initial_metrics[split] = evaluate(
            views[split], gray, gaussians, pipe, background, dataset, device, beta,
            output / "evaluation_initial" / split, args.save_images, radiometric_response, loss_config=shared, override_opacity=override_opacity)
    append_log(log, {"step": 0, "initial_metrics": initial_metrics})
    best_validation, best_step = initial_metrics["validation"]["radiance_loss"], 0
    save_control(output / "gray_best.pt", gray, 0, best_validation, protocol)
    optimizer = torch.optim.Adam([gray], lr=args.gray_lr)
    cameras = views["fit"]
    order, rng = list(range(len(cameras))), random.Random(args.seed)
    for step in range(1, args.steps + 1):
        position = (step - 1) % len(cameras)
        if position == 0:
            rng.shuffle(order)
        lr = held_lr(step, args.steps, args.gray_lr, args.gray_lr_final, args.lr_hold_fraction)
        optimizer.param_groups[0]["lr"] = lr
        camera = cameras[order[position]]
        prediction = prediction_image(camera, gray, gaussians, pipe, background, dataset, device, override_opacity)
        target = target_image(camera, device)
        if prediction.shape != target.shape:
            raise ValueError("Image dimensions differ: " + camera.image_name)
        loss = thermal_data_loss(prediction, target, shared, getattr(camera, "thermal_valid_mask", None))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gray.grad is None or not bool(torch.isfinite(gray.grad).all()):
            raise FloatingPointError("Missing or non-finite grayscale gradient")
        optimizer.step()
        with torch.no_grad():
            gray.clamp_(0, 1)
        if step == 1 or step % 100 == 0 or step == args.steps:
            append_log(log, {"step": step, "gray_lr": lr, "radiance_loss": float(loss.detach())})
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(views["validation"], gray, gaussians, pipe, background,
                               dataset, device, beta, radiometric_response=radiometric_response, loss_config=shared, override_opacity=override_opacity)
            if metrics["radiance_loss"] < best_validation:
                best_validation, best_step = metrics["radiance_loss"], step
                save_control(output / "gray_best.pt", gray, step, best_validation, protocol)
            append_log(log, {"step": step, "internal_validation": metrics,
                             "best_step": best_step, "best_validation_Huber": best_validation})
        if step % args.save_every == 0:
            save_control(output / ("gray_step_%d.pt" % step), gray, step, None, protocol)
    save_control(output / "gray_final.pt", gray, args.steps, None, protocol)
    report = {"protocol": protocol, "best_step": best_step,
              "initial": initial_metrics, "final": {}, "best": {}}
    # Test never participates in optimization, checkpoint selection, or stopping.
    for split in ("fit", "validation"):
        report["final"][split] = evaluate(
            views[split], gray, gaussians, pipe, background, dataset, device, beta,
            output / "evaluation_final" / split, args.save_images, radiometric_response, loss_config=shared, override_opacity=override_opacity)
    best = torch.load(output / "gray_best.pt", map_location=device)["gray"]
    for split in ("fit", "validation", "test"):
        report["best"][split] = evaluate(
            views[split], best, gaussians, pipe, background, dataset, device, beta,
            output / "evaluation_best" / split, args.save_images, radiometric_response, loss_config=shared, override_opacity=override_opacity)
        print("[best/%s] %s" % (split, json.dumps(report["best"][split])), flush=True)
    report["validation_change_from_stage2"] = {
        key: report["best"]["validation"][key] - initial_metrics["validation"][key]
        for key in ("PSNR", "SSIM", "radiance_loss")
    }
    if args.observation_domain == "raw_rjpeg":
        # Recoloring can reuse the reference checkpoint's frozen display map.
        for phase in ("initial", "final", "best"):
            phase_report = {"modality": "thermal", "observation_domain": "raw_rjpeg",
                "metric_domain": "full_frame_float_normalized_camera_signal",
                "radiometric_protocol": args.radiometric_protocol,
                "loss_scale_protocol": shared.get("loss_scale_protocol"),
                "kind": "free_gray_control", "splits": report[phase]}
            if args.radiometric_protocol["format_version"] == 2:
                phase_report.update(metric_domain="valid_native_FOV_float_normalized_camera_signal",
                    signal_metric_region="valid_native_FOV", signal_SSIM_region="full_frame")
            write_json(output / ("evaluation_" + phase) / "metrics.json", phase_report)
    write_json(output / "metrics.json", report)
    print("[done] " + str(output), flush=True)


if __name__ == "__main__":
    main()
