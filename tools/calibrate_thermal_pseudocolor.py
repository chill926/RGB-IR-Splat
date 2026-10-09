"""Freeze a train-only signal-to-original-pseudocolor mapping (CPU is enough)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.thermal_pseudocolor import calibrate_display, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Stage2 reference supplies the exact fit/validation/test split")
    parser.add_argument("--radiometric_dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--sample_stride", type=int, default=4)
    parser.add_argument("--allow_new_radiometric", action="store_true",
                        help="Use reference split only for freshly aligned inputs with unchanged camera response")
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    shared = metadata.get("shared_training_protocol", {})
    if shared.get("observation_domain") != "raw_rjpeg" or not isinstance(shared.get("radiometric_protocol"), dict):
        raise ValueError("Display calibration requires a raw_rjpeg thermal checkpoint")
    split = metadata.get("camera_split", {})
    if not split.get("fit") or not split.get("validation") or not split.get("test"):
        raise ValueError("Checkpoint must record all three camera splits")
    directory = args.radiometric_dir or shared.get("radiometric_dir", "radiometric")
    expected = shared["radiometric_protocol"]
    if args.allow_new_radiometric:
        _, manifest, digest = load_manifest(args.scene, directory)
        if (manifest["signal_calibration"] != expected["signal_calibration"] or
                manifest.get("coordinate_alignment", {}).get("camera_split") != split):
            raise ValueError("Fresh display mapping requires unchanged response and exact reference alignment split")
        expected = {"manifest_sha256": digest, "signal_calibration": manifest["signal_calibration"]}
    document = calibrate_display(args.scene, directory, split["fit"], args.output,
                                 camera_split=split, expected_protocol=expected,
                                 bins=args.bins, stride=args.sample_stride)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "fit_views": len(document["fit_camera_names"]),
                      "palettes": len(document["groups"]),
                      "mapping": document["display_model"],
                      "validation_or_test_pixels_used_for_fitting": False}, indent=2))


if __name__ == "__main__":
    main()
