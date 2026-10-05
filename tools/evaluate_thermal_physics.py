"""Evaluate frozen thermal fields or their stage-1 RGB geometry and appearance.

Install this file in graduate/tools/. The default split is the exact internal
validation split recorded by stage 2; no model parameters are optimized.
RGB mode uses the thermal checkpoint only to identify geometry and camera splits.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from train_thermal_physics import (
    geometry_checkpoint_sha256,
    observation_to_model_radiance,
)
from utils.loss_utils import ssim
from utils.thermal_physics import (
    MaterialThermalField,
    UniformLWIRPlanckLUT,
    verify_frozen_geometry_state,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_views(scene, metadata, split):
    published_train = scene.getTrainCameras()
    published_test = scene.getTestCameras()
    available = published_test if split == "test" else published_train
    names = metadata.get("camera_split", {}).get(split)
    if not names:
        raise ValueError(f"Checkpoint has no recorded {split} cameras")
    by_name = {view.image_name: view for view in available}
    if len(by_name) != len(available) or len(set(names)) != len(names):
        raise ValueError("Duplicate camera names prevent unambiguous evaluation")
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Recorded {split} cameras are missing: {missing[:8]}")
    return [by_name[name] for name in names]


@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    model_params = ModelParams(parser, device)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", choices=("thermal", "rgb"), default="thermal",
                        help="RGB evaluates stage-1 RGB SH; the checkpoint supplies geometry and view names")
    parser.add_argument("--geometry_model", default="",
                        help="Optional override when the recorded geometry directory was moved")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--split", choices=("fit", "validation", "test", "all"),
                        default="validation")
    parser.add_argument("--save_images", action="store_true",
                        help="Save float arrays, PNGs, and GT/prediction/error comparisons")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation needs the same CUDA/3DGS environment used for training")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    branch = checkpoint.get("branch")
    if branch not in MaterialThermalField.BRANCHES:
        raise ValueError(f"Unsupported thermal checkpoint branch: {branch}")
    shared = metadata.get("shared_training_protocol", {})
    if "observation_domain" not in shared:
        raise ValueError("Checkpoint is missing its observation-domain protocol")
    observation_args = SimpleNamespace(**shared)
    args.source_path = os.path.abspath(args.source_path or metadata.get("source_path", ""))
    geometry_model = args.geometry_model or metadata.get("geometry_model", "")
    if not geometry_model:
        raise ValueError("No geometry directory recorded; pass --geometry_model")
    geometry_model = os.path.abspath(geometry_model)
    iteration = metadata.get("geometry_iteration")
    if iteration is None or int(iteration) <= 0:
        raise ValueError("Checkpoint must record an explicit positive geometry iteration")
    expected_hash = metadata.get("geometry_sha256")
    if not expected_hash or geometry_checkpoint_sha256(geometry_model, int(iteration)) != expected_hash:
        raise ValueError("Geometry PLY does not match the checkpoint's recorded geometry hash")
    output_root = Path(args.output_dir).resolve() if args.output_dir else (
        checkpoint_path.parent / f"evaluation_{args.modality}_{checkpoint_path.stem}")
    # Scene loads the fixed geometry, while all evaluation artifacts go here.
    args.model_path = str(output_root)
    args.load_model_path = geometry_model
    args.data_branch = "rgbt"
    args.eval = True
    dataset = model_params.extract(args)
    if args.modality == "rgb":
        dataset.rgb_geometry_stage = True
    pipeline = pipeline_params.extract(args)
    output_root.mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians, load_iteration=int(iteration), shuffle=False)
    if not scene.has_rgbt:
        raise ValueError("Evaluation requires an RGBT scene")
    if checkpoint.get("frozen_geometry") is not None:
        verify_frozen_geometry_state(gaussians, checkpoint["frozen_geometry"])
    planck, colors = None, None
    if args.modality == "thermal":
        field = MaterialThermalField.from_checkpoint(
            checkpoint, branch, device, reset_branch_parameters=False).eval()
        if field.temperature.shape[0] != gaussians.get_xyz.shape[0]:
            raise ValueError("Thermal field and geometry have different Gaussian counts")
        config = checkpoint["config"]
        planck = UniformLWIRPlanckLUT(config["temp_min"], config["temp_max"]).to(device)
        colors = field.radiance(planck).repeat(1, 3)
    background = torch.zeros(3, device=device)
    splits = ("fit", "validation", "test") if args.split == "all" else (args.split,)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "branch": branch,
        "modality": args.modality,
        "evaluated_stage": "stage1_rgb" if args.modality == "rgb" else branch,
        "source_path": args.source_path,
        "geometry_model": geometry_model,
        "geometry_iteration": int(iteration),
        "geometry_sha256": expected_hash,
        "observation_domain": "rgb" if args.modality == "rgb" else shared["observation_domain"],
        "metric_domain": ("full_frame_float_rgb" if args.modality == "rgb"
                          else "full_frame_float_normalized_radiance"),
        "metric_data_range": 1.0,
        "psnr_mse_floor": 1e-12,
        "ssim_window_size": 11,
        "comparison_order": ["ground_truth", "prediction", "absolute_error"],
        "splits": {},
    }
    for split in splits:
        views = select_views(scene, metadata, split)
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        if args.save_images:
            for folder in ("renders", "gt", "comparisons", "float_arrays"):
                (split_root / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for view in tqdm(views, desc=f"Evaluating {args.modality} ({split})"):
            prediction = render(
                view, gaussians, pipeline, background, 0.0, 0.0, 0.0,
                device, dataset.is_6dof, override_color=colors,
                feature_set="rgb", detach_geometry=True)["render"]
            if args.modality == "rgb":
                if view.original_rgb_image is None:
                    raise ValueError(f"Missing RGB target: {view.image_name}")
                prediction = prediction[:3].clamp(0.0, 1.0)
                target = view.original_rgb_image[:3].clamp(0.0, 1.0)
            else:
                prediction = prediction[:1]
                source = (view.original_physical_image if view.original_physical_image is not None
                          else view.original_image)
                target = observation_to_model_radiance(
                    source.mean(dim=0, keepdim=True), observation_args, planck)
            if prediction.shape != target.shape:
                raise ValueError(f"Prediction/target dimensions differ: {view.image_name}")
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise FloatingPointError(f"Non-finite image: {view.image_name}")
            error = prediction - target
            mse_value = float(error.square().mean())
            row = {
                "image_name": view.image_name,
                "PSNR": -10.0 * math.log10(max(mse_value, 1e-12)),
                "SSIM": float(ssim(prediction[None], target[None])),
                "MSE": mse_value,
                "RMSE": math.sqrt(mse_value),
                "MAE": float(error.abs().mean()),
            }
            if args.modality == "thermal":
                row["radiance_loss"] = float(F.smooth_l1_loss(
                    prediction, target, beta=shared.get("huber_delta", 0.02)))
            rows.append(row)
            if args.save_images:
                name = view.image_name
                save_image(prediction.clamp(0, 1), split_root / "renders" / f"{name}.png")
                save_image(target.clamp(0, 1), split_root / "gt" / f"{name}.png")
                comparison = torch.cat((target, prediction, error.abs()), dim=-1).clamp(0, 1)
                save_image(comparison, split_root / "comparisons" / f"{name}.png")
                np.save(split_root / "float_arrays" / f"{name}.prediction.npy",
                        prediction.detach().cpu().numpy())
                np.save(split_root / "float_arrays" / f"{name}.gt.npy",
                        target.detach().cpu().numpy())
        metric_keys = ["PSNR", "SSIM", "MSE", "MAE"]
        if args.modality == "thermal":
            metric_keys.append("radiance_loss")
        summary = {key: sum(row[key] for row in rows) / len(rows) for key in metric_keys}
        summary["RMSE"] = math.sqrt(summary["MSE"])
        summary["views"] = len(rows)
        with (split_root / "per_view.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False, allow_nan=False)
        report["splits"][split] = summary
        print(f"[{split}] " + json.dumps(summary, allow_nan=False))
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[evaluation] wrote {output_root}")


if __name__ == "__main__":
    main()
