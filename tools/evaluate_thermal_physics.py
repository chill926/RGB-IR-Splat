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
from utils.thermal_opacity import IROpacityCorrection, render_opacity
from scene import GaussianModel, Scene
from train_thermal_physics import (
    geometry_checkpoint_sha256,
    observation_to_model_radiance,
)
from utils.loss_utils import ssim
from utils.thermal_training_utils import thermal_data_loss
from utils.thermal_observations import prepare_thermal_input, radiometric_image_errors
from utils.thermal_pseudocolor import (FrozenDisplay, evaluate_pseudocolor,
    load_manifest, thermal_reference, original_rgb)
from utils.thermal_physics import (
    MaterialThermalField,
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
    parser.add_argument("--radiometric_dir", default="",
                        help="Optional moved RJPEG signal directory; content must match checkpoint")
    parser.add_argument("--thermal_output", choices=("signal", "pseudocolor"), default="signal",
                        help="Pseudo-color renders are restored after signal alpha blending and compared to original thermal RGB")
    parser.add_argument("--display_calibration", default="",
                        help="Frozen fit-only display calibration from tools/calibrate_thermal_pseudocolor.py")
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
    if shared["observation_domain"] == "raw_rjpeg" and not isinstance(shared.get("radiometric_protocol"), dict):
        raise ValueError("RJPEG checkpoint is missing its radiometric calibration protocol")
    observation_args = SimpleNamespace(**shared)
    use_pseudocolor = args.thermal_output == "pseudocolor"
    if use_pseudocolor and (args.modality != "thermal" or shared["observation_domain"] != "raw_rjpeg"):
        raise ValueError("Pseudo-color output requires --modality thermal and a raw_rjpeg checkpoint")
    if use_pseudocolor and not args.display_calibration:
        raise ValueError("--display_calibration is required for pseudo-color output")
    args.source_path = os.path.abspath(args.source_path or metadata.get("source_path", ""))
    observation_args.source_path = args.source_path
    if args.radiometric_dir:
        observation_args.radiometric_dir = args.radiometric_dir
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
    planck, colors, opacity_field = None, None, None
    if args.modality == "thermal":
        field = MaterialThermalField.from_checkpoint(
            checkpoint, branch, device, reset_branch_parameters=False).eval()
        if field.temperature.shape[0] != gaussians.get_xyz.shape[0]:
            raise ValueError("Thermal field and geometry have different Gaussian counts")
        config = checkpoint["config"]
        observation_args.temp_min, observation_args.temp_max = config["temp_min"], config["temp_max"]
        planck = prepare_thermal_input(
            observation_args, scene.getTrainCameras() + scene.getTestCameras(), device)
        colors = field.radiance(planck).repeat(1, 3)
        opacity_field = IROpacityCorrection.from_checkpoint(checkpoint, gaussians.get_opacity, trainable=False)
    background = torch.zeros(3, device=device)
    display, display_manifest = None, None
    if use_pseudocolor:
        display = FrozenDisplay(args.display_calibration, shared["radiometric_protocol"],
                                metadata.get("camera_split", {}).get("fit"))
        _, display_manifest, _ = load_manifest(args.source_path, observation_args.radiometric_dir,
                                               shared["radiometric_protocol"])
    splits = ("fit", "validation", "test") if args.split == "all" else (args.split,)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "branch": branch,
        "training_stage": metadata.get("training_stage", branch),
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
        "ir_opacity_enabled": opacity_field is not None,
        "ir_opacity_diagnostics": opacity_field.diagnostics() if opacity_field is not None else None,
        "splits": {},
    }
    if args.modality == "thermal" and shared["observation_domain"] == "raw_rjpeg":
        report["radiometric_protocol"] = observation_args.radiometric_protocol
        report["loss_scale_protocol"] = shared.get("loss_scale_protocol")
        report["metric_domain"] = "full_frame_float_normalized_camera_signal"
        report["apparent_temperature_error_semantics"] = (
            "Blackbody-equivalent observation error at the camera, not surface-temperature GT error")
    if use_pseudocolor:
        report.update({"thermal_output": "pseudocolor", "metric_domain": "original_dataset_pseudocolor_rgb",
                       "signal_metric_domain": "full_frame_float_normalized_camera_signal",
                       "display_calibration": str(Path(args.display_calibration).resolve()),
                       "display_calibration_sha256": display.sha256,
                       "display_mapping_exact_camera_agc": False,
                       "float_array_domain": "normalized_camera_signal; pseudo-color arrays have explicit pseudocolor suffix",
                       "roundtrip_comparison_order": ["original_ground_truth", "signal_roundtrip", "absolute_error"]})
    for split in splits:
        views = select_views(scene, metadata, split)
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        if args.save_images:
            for folder in ("renders", "gt", "comparisons", "float_arrays"):
                (split_root / folder).mkdir(parents=True, exist_ok=True)
            if use_pseudocolor:
                for folder in ("roundtrip", "roundtrip_comparisons", "signal_renders", "signal_gt"):
                    (split_root / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for view in tqdm(views, desc=f"Evaluating {args.modality} ({split})"):
            prediction = render(
                view, gaussians, pipeline, background, 0.0, 0.0, 0.0,
                device, dataset.is_6dof, override_color=colors,
                feature_set="rgb", detach_geometry=True,
                override_opacity=render_opacity(opacity_field))["render"]
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
                row["radiance_loss"] = float(thermal_data_loss(prediction, target, shared))
                row["normalized_signal_huber_loss"] = float(F.smooth_l1_loss(
                    prediction, target, beta=shared.get("huber_delta", 0.02)))
                if shared["observation_domain"] == "raw_rjpeg":
                    row.update(radiometric_image_errors(prediction, target, planck))
            if use_pseudocolor:
                name = view.image_name
                record = display_manifest["frames"][name]
                display.verify_reference(name, record)
                thermal_reference(args.source_path, name, record)
                signal_prediction, signal_target = prediction, target
                # RGBT Camera.original_image is grayscale, not the dataset palette RGB.
                original = original_rgb(args.source_path, name, record, shape=prediction.shape[-2:])
                rgb, roundtrip, restored_metrics = evaluate_pseudocolor(
                    display, prediction.detach().cpu().numpy(), target.detach().cpu().numpy(),
                    original, name, [config["temp_min"], config["temp_max"]])
                core_keys = ("PSNR", "SSIM", "MSE", "RMSE", "MAE")
                row = {"image_name": name, **restored_metrics,
                       **{"signal_" + key: row[key] for key in core_keys},
                       **{key: value for key, value in row.items() if key not in core_keys and key != "image_name"}}
                prediction = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device)
                target = torch.from_numpy(original.transpose(2, 0, 1).copy()).to(device)
                error = prediction - target
            rows.append(row)
            if args.save_images:
                name = view.image_name
                save_image(prediction.clamp(0, 1), split_root / "renders" / f"{name}.png")
                save_image(target.clamp(0, 1), split_root / "gt" / f"{name}.png")
                comparison = torch.cat((target, prediction, error.abs()), dim=-1).clamp(0, 1)
                save_image(comparison, split_root / "comparisons" / f"{name}.png")
                np.save(split_root / "float_arrays" / f"{name}.prediction.npy",
                        (signal_prediction if use_pseudocolor else prediction).detach().cpu().numpy())
                np.save(split_root / "float_arrays" / f"{name}.gt.npy",
                        (signal_target if use_pseudocolor else target).detach().cpu().numpy())
                if use_pseudocolor:
                    roundtrip_tensor = torch.from_numpy(roundtrip.transpose(2, 0, 1).copy()).to(device)
                    save_image(roundtrip_tensor, split_root / "roundtrip" / f"{name}.png")
                    save_image(torch.cat((target, roundtrip_tensor, (roundtrip_tensor - target).abs()), dim=-1).clamp(0, 1),
                               split_root / "roundtrip_comparisons" / f"{name}.png")
                    save_image(signal_prediction, split_root / "signal_renders" / f"{name}.png")
                    save_image(signal_target, split_root / "signal_gt" / f"{name}.png")
                    np.save(split_root / "float_arrays" / f"{name}.pseudocolor_prediction.npy", rgb)
                    np.save(split_root / "float_arrays" / f"{name}.pseudocolor_gt.npy", original)
        metric_keys = ["PSNR", "SSIM", "MSE", "MAE"]
        if args.modality == "thermal":
            metric_keys.extend(("radiance_loss", "normalized_signal_huber_loss"))
            if shared["observation_domain"] == "raw_rjpeg":
                metric_keys.extend(("camera_signal_MAE", "camera_signal_RMSE",
                                    "apparent_temperature_MAE_K"))
        if use_pseudocolor:
            metric_keys.extend(("signal_PSNR", "signal_SSIM", "signal_MSE", "signal_RMSE", "signal_MAE",
                                "mapping_roundtrip_PSNR", "mapping_roundtrip_SSIM", "mapping_roundtrip_MSE",
                                "mapping_roundtrip_RMSE", "mapping_roundtrip_MAE",
                                "prediction_outside_display_support_fraction",
                                "observation_outside_display_support_fraction"))
        summary = {key: sum(row[key] for row in rows) / len(rows) for key in metric_keys}
        summary["RMSE"] = math.sqrt(summary["MSE"])
        summary["views"] = len(rows)
        if args.modality == "thermal" and shared["observation_domain"] == "raw_rjpeg":
            signal_mse = summary["signal_MSE"] if use_pseudocolor else summary["MSE"]
            span = float(planck.physical_radiance_max - planck.physical_radiance_min)
            summary["camera_signal_RMSE"] = math.sqrt(signal_mse) * span
        if use_pseudocolor:
            summary["mapping_roundtrip_RMSE"] = math.sqrt(summary["mapping_roundtrip_MSE"])
            summary["signal_RMSE"] = math.sqrt(summary["signal_MSE"])
        with (split_root / "per_view.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False, allow_nan=False)
        report["splits"][split] = summary
        print(f"[{split}] " + json.dumps(summary, allow_nan=False))
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[evaluation] wrote {output_root}")


if __name__ == "__main__":
    main()
