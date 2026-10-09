"""Estimate a fit-only coordinate correction, then prepare a separate signal version."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.thermal_coordinates import fit_alignment, profile_sha256
from tools.prepare_rgbt_radiometry import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Existing raw_rjpeg Stage2 checkpoint: split/config only")
    parser.add_argument("--scene", default="")
    parser.add_argument("--output_dir", required=True, help="New empty radiometric directory; never overwrite old inputs")
    parser.add_argument("--alignment_output", required=True, help="New JSON file for the frozen fit-only profile")
    parser.add_argument("--sample_stride", type=int, default=4)
    parser.add_argument("--max_shift_pixels", type=float, default=8.)
    parser.add_argument("--byte_order", choices=("auto", "native", "swap"), default="auto")
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(args.reference, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    shared = metadata.get("shared_training_protocol", {})
    if checkpoint.get("branch") != "stage2" or shared.get("observation_domain") != "raw_rjpeg":
        raise ValueError("Reference must be a raw_rjpeg Stage2 checkpoint")
    if not isinstance(shared.get("radiometric_protocol"), dict):
        raise ValueError("Reference has no radiometric protocol")
    scene = Path(args.scene or metadata["source_path"]).resolve()
    output, profile_path = Path(args.output_dir).resolve(), Path(args.alignment_output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Radiometric output is not empty; choose a new version")
    if profile_path.exists():
        raise ValueError("Alignment output already exists; reuse it with prepare_rgbt_radiometry.py --alignment")
    if output == profile_path.parent or output in profile_path.parents:
        raise ValueError("Alignment JSON must be outside the initially empty radiometric directory")
    for protected in (scene / "raw_images", scene / "rgb", scene / "thermal", Path(args.reference).resolve().parent):
        if profile_path == protected or protected in profile_path.parents:
            raise ValueError("Alignment profile must not be written into existing input/model directories")
    def progress(current, total):
        if current == 1 or current % 8 == 0 or current == total:
            print("[alignment] fit %d/%d" % (current, total), flush=True)
    profile = fit_alignment(scene, metadata["camera_split"], args.sample_stride, args.max_shift_pixels,
                            args.byte_order, progress)
    print("[alignment] " + json.dumps({"offset_reference_pixels": profile["offset_reference_pixels"],
        "accepted_fit_frames": profile["diagnostics"]["accepted_frames"], "MAD_pixels": profile["diagnostics"]["MAD_pixels"],
        "sampling": "output (x,y) samples old full-frame signal (x+dx,y+dy)",
        "validation_or_test_pixels_used_for_fitting": False}), flush=True)
    manifest = prepare(scene, output, args.byte_order, alignment=profile)
    if manifest["signal_calibration"] != shared["radiometric_protocol"]["signal_calibration"]:
        raise ValueError("Prepared camera response differs from the reference; inspect the new inputs")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"alignment_output": str(profile_path), "alignment_sha256": profile_sha256(profile),
        "radiometric_dir": str(output), "frames": len(manifest["frames"]),
        "valid_pixel_fraction_min": min(r["valid_pixel_fraction"] for r in manifest["frames"].values()),
        "old_Stage2_weights_used": False, "old_data_overwritten": False}, indent=2))


if __name__ == "__main__":
    main()
