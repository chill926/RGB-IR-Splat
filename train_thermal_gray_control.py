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
from scene import GaussianModel, Scene
from utils.loss_utils import ssim
from utils.thermal_physics import (
    MaterialThermalField, UniformLWIRPlanckLUT, verify_frozen_geometry_state,
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


def prediction_image(camera, gray, gaussians, pipe, background, dataset, device):
    prediction = render(
        camera, gaussians, pipe, background, 0.0, 0.0, 0.0, device,
        dataset.is_6dof, override_color=gray.repeat(1, 3),
        detach_geometry=True)["render"][:1]
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Non-finite prediction: " + camera.image_name)
    return prediction


@torch.no_grad()
def evaluate(views, gray, gaussians, pipe, background, dataset, device, beta,
             directory=None, save_images=False):
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        if save_images:
            for folder in ("renders", "gt", "comparisons"):
                (directory / folder).mkdir(exist_ok=True)
    rows = []
    for camera in tqdm(views, desc="Evaluate", leave=False):
        prediction = prediction_image(camera, gray, gaussians, pipe, background, dataset, device)
        target = target_image(camera, device)
        if prediction.shape != target.shape:
            raise ValueError("Image dimensions differ: " + camera.image_name)
        error = prediction - target
        mse = float(error.square().mean())
        rows.append({
            "image_name": camera.image_name,
            "PSNR": -10.0 * math.log10(max(mse, 1e-12)),
            "SSIM": float(ssim(prediction[None], target[None])),
            "MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(error.abs().mean()),
            "radiance_loss": float(F.smooth_l1_loss(prediction, target, beta=beta)),
        })
        if directory is not None and save_images:
            filename = camera.image_name + ".png"
            save_image(prediction.clamp(0, 1), directory / "renders" / filename)
            save_image(target, directory / "gt" / filename)
            save_image(torch.cat((target, prediction, error.abs()), dim=-1).clamp(0, 1),
                       directory / "comparisons" / filename)
    keys = ("PSNR", "SSIM", "MSE", "MAE", "radiance_loss")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    summary.update(RMSE=math.sqrt(summary["MSE"]), views=len(rows))
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
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--gray_lr", type=float, default=1e-2)
    parser.add_argument("--gray_lr_final", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the CUDA server in the physir environment")
    if min(args.steps, args.eval_every, args.save_every) <= 0:
        raise ValueError("Step counts and evaluation/save intervals must be positive")
    if not 0 < args.gray_lr_final <= args.gray_lr or not math.isfinite(args.gray_lr):
        raise ValueError("Require finite 0 < gray_lr_final <= gray_lr")
    if not args.model_path:
        raise ValueError("Use -m to specify a new, separate output directory")
    checkpoint_path = Path(args.stage2_checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("branch") != "stage2":
        raise ValueError("Reference must be a stage2 checkpoint, not C/K/R")
    metadata = checkpoint.get("metadata", {})
    shared = metadata.get("shared_training_protocol", {})
    if shared.get("observation_domain") != "normalized_dn":
        raise ValueError("This diagnostic supports the current normalized_dn experiment only")
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
        planck = UniformLWIRPlanckLUT(config["temp_min"], config["temp_max"]).to(device)
        initial_gray = field.radiance(planck).detach().clone()
    if initial_gray.shape != (gaussians.get_xyz.shape[0], 1):
        raise ValueError("Reference field and geometry have different Gaussian counts")
    if not bool(torch.isfinite(initial_gray).all()):
        raise FloatingPointError("Reference initial radiance is non-finite")
    del field, planck, checkpoint
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
        "resolution": args.resolution, "observation_domain": "normalized_dn",
        "loss": "Huber only", "huber_delta": beta, "gray_bounds": [0, 1],
        "initialization": "per_Gaussian_stage2_final_radiance",
        "selection": "minimum_internal_validation_Huber_including_initialization",
        "metric_data_range": 1.0, "ssim_window_size": 11,
        "comparison_order": ["ground_truth", "prediction", "absolute_error"],
    }
    write_json(output / "protocol.json", protocol)
    print("[split] fit=%d, internal_validation=%d, published_test=%d" %
          (len(views["fit"]), len(views["validation"]), len(views["test"])), flush=True)
    log = output / "gray_training.jsonl"
    initial_metrics = {}
    for split in ("fit", "validation"):
        initial_metrics[split] = evaluate(
            views[split], gray, gaussians, pipe, background, dataset, device, beta,
            output / "evaluation_initial" / split, args.save_images)
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
        lr = math.exp(math.log(args.gray_lr) * (1 - step / args.steps) +
                      math.log(args.gray_lr_final) * (step / args.steps))
        optimizer.param_groups[0]["lr"] = lr
        camera = cameras[order[position]]
        prediction = prediction_image(camera, gray, gaussians, pipe, background, dataset, device)
        target = target_image(camera, device)
        if prediction.shape != target.shape:
            raise ValueError("Image dimensions differ: " + camera.image_name)
        loss = F.smooth_l1_loss(prediction, target, beta=beta)
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
                               dataset, device, beta)
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
            output / "evaluation_final" / split, args.save_images)
    best = torch.load(output / "gray_best.pt", map_location=device)["gray"]
    for split in ("fit", "validation", "test"):
        report["best"][split] = evaluate(
            views[split], best, gaussians, pipe, background, dataset, device, beta,
            output / "evaluation_best" / split, args.save_images)
        print("[best/%s] %s" % (split, json.dumps(report["best"][split])), flush=True)
    report["validation_change_from_stage2"] = {
        key: report["best"]["validation"][key] - initial_metrics["validation"][key]
        for key in ("PSNR", "SSIM", "radiance_loss")
    }
    write_json(output / "metrics.json", report)
    print("[done] " + str(output), flush=True)


if __name__ == "__main__":
    main()
