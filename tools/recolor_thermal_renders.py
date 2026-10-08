"""Restore existing float signal renders to original thermal pseudocolor, on CPU.

No CUDA, 3DGS extension, or checkpoint loading is needed. Input arrays must come
from evaluate_thermal_physics.py --save_images in raw_rjpeg mode.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from utils.thermal_pseudocolor import (FrozenDisplay, evaluate_pseudocolor, load_manifest,
    original_rgb, save_rgb)


def summarize(rows):
    keys = [key for key in rows[0] if key != "image_name" and isinstance(rows[0][key], (int, float))]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["RMSE"] = math.sqrt(summary["MSE"])
    if "signal_MSE" in summary:
        summary["signal_RMSE"] = math.sqrt(summary["signal_MSE"])
    summary["mapping_roundtrip_RMSE"] = math.sqrt(summary["mapping_roundtrip_MSE"])
    if "camera_signal_RMSE" in summary:
        summary["camera_signal_RMSE"] = math.sqrt(np.mean([row["camera_signal_RMSE"] ** 2 for row in rows]))
    summary["views"] = len(rows)
    return summary


def recolor(scene, evaluation_dir, calibration, output_dir, split="validation", radiometric_dir="radiometric"):
    source, output = Path(evaluation_dir).resolve(), Path(output_dir).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Choose a separate output directory for pseudo-color results")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Pseudo-color output is not empty; choose a new --output_dir")
    report = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    protocol = report.get("radiometric_protocol")
    if report.get("modality") != "thermal" or report.get("observation_domain") != "raw_rjpeg" or not protocol:
        raise ValueError("Source evaluation must contain normalized raw_rjpeg thermal renders")
    if report.get("metric_domain") != "full_frame_float_normalized_camera_signal":
        raise ValueError("Recoloring expects original signal-domain evaluation, not already colorized images")
    _, manifest, _ = load_manifest(scene, radiometric_dir, protocol)
    display = FrozenDisplay(calibration, protocol)
    recorded = display.document.get("reference_camera_split")
    if not recorded:
        raise ValueError("Display calibration must record its checkpoint camera split")
    bounds = protocol["normalization_temperature_bounds_K"]
    splits = tuple(report["splits"]) if split == "all" else (split,)
    plans = {}
    # Validate split membership before writing any output. Held-out labels are
    # read only for evaluation after calibration is frozen.
    for role in splits:
        if role not in report["splits"] or role not in recorded:
            raise ValueError("Requested split is not present in the source evaluation/calibration: " + role)
        rows = json.loads((source / role / "per_view.json").read_text(encoding="utf-8"))
        if not rows or {row["image_name"] for row in rows} != set(recorded[role]):
            raise ValueError("Source evaluation does not match the calibration camera split: " + role)
        if len({row["image_name"] for row in rows}) != len(rows):
            raise ValueError("Duplicate source evaluation image names")
        plans[role] = rows
    result = {key: value for key, value in report.items() if key != "splits"}
    result.update({"metric_domain": "original_dataset_pseudocolor_rgb",
                   "thermal_output": "pseudocolor", "metric_data_range": 1.0,
                   "signal_metric_domain": report["metric_domain"],
                   "display_calibration": str(Path(calibration).resolve()),
                   "display_calibration_sha256": display.sha256,
                   "display_mapping_exact_camera_agc": False,
                   "source_signal_evaluation": str(source),
                   "roundtrip_comparison_order": ["original_ground_truth", "signal_roundtrip", "absolute_error"],
                   "splits": {}})
    output.mkdir(parents=True, exist_ok=True)
    for role, previous_rows in plans.items():
        root = output / role
        for folder in ("renders", "gt", "comparisons", "roundtrip", "roundtrip_comparisons", "float_arrays"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for old in previous_rows:
            name = old["image_name"]
            if name not in manifest["frames"]:
                raise ValueError("Unknown radiometric frame: " + name)
            record = manifest["frames"][name]
            if record["split"] != ("test" if role == "test" else "train"):
                raise ValueError("Camera split differs from prepared data: " + name)
            display.verify_reference(name, record)
            prediction = np.load(source / role / "float_arrays" / f"{name}.prediction.npy", allow_pickle=False)
            target = np.load(source / role / "float_arrays" / f"{name}.gt.npy", allow_pickle=False)
            if prediction.shape != target.shape:
                raise ValueError("Signal prediction/target dimensions differ: " + name)
            original = original_rgb(scene, name, record, shape=prediction.shape[-2:])
            rgb, roundtrip, metrics = evaluate_pseudocolor(display, prediction, target, original, name, bounds)
            row = {"image_name": name, **metrics}
            for key, value in old.items():
                if key in ("PSNR", "SSIM", "MSE", "RMSE", "MAE"):
                    row["signal_" + key] = value
                elif key != "image_name":
                    row[key] = value
            rows.append(row)
            save_rgb(root / "renders" / f"{name}.png", rgb)
            save_rgb(root / "gt" / f"{name}.png", original)
            save_rgb(root / "comparisons" / f"{name}.png", np.concatenate((original, rgb, np.abs(rgb - original)), axis=1))
            save_rgb(root / "roundtrip" / f"{name}.png", roundtrip)
            save_rgb(root / "roundtrip_comparisons" / f"{name}.png", np.concatenate((original, roundtrip, np.abs(roundtrip - original)), axis=1))
            np.save(root / "float_arrays" / f"{name}.pseudocolor_prediction.npy", rgb)
            np.save(root / "float_arrays" / f"{name}.pseudocolor_gt.npy", original)
        (root / "per_view.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
        summary = summarize(rows)
        result["splits"][role] = summary
        print(f"[{role} pseudocolor] " + json.dumps(summary, allow_nan=False))
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--display_calibration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--radiometric_dir", default="radiometric")
    parser.add_argument("--split", choices=("fit", "validation", "test", "all"), default="validation")
    args = parser.parse_args()
    recolor(args.scene, args.evaluation_dir, args.display_calibration, args.output_dir,
            args.split, args.radiometric_dir)


if __name__ == "__main__":
    main()
