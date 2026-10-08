"""Warm-start bounded IR-opacity adaptation and evaluate a fixed-budget result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Stage2 common checkpoint")
    parser.add_argument("--output", required=True, help="New, empty output directory")
    parser.add_argument("--mode", choices=("bounded", "frozen"), default="bounded",
                        help="Use frozen for a matched G0 warm-start/TV/LR control")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lambda_tv", type=float, default=0.001)
    parser.add_argument("--opacity_bound", type=float, default=0.2)
    parser.add_argument("--opacity_lr", type=float, default=1e-3)
    parser.add_argument("--opacity_lr_final", type=float, default=1e-5)
    parser.add_argument("--lambda_opacity", type=float, default=0.01)
    parser.add_argument("--temperature_lr", type=float, default=1e-4)
    parser.add_argument("--temperature_lr_final", type=float, default=1e-6)
    parser.add_argument("--scene", default="")
    parser.add_argument("--geometry_model", default="")
    parser.add_argument("--radiometric_dir", default="")
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(args.reference, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    if checkpoint.get("branch") != "stage2" or not metadata.get("stage2_common_endpoint"):
        raise ValueError("Reference must be a Stage2 common checkpoint")
    shared = metadata["shared_training_protocol"]
    if shared.get("observation_domain") != "raw_rjpeg":
        raise ValueError("This runner evaluates original pseudocolor and requires raw_rjpeg")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from utils.thermal_pseudocolor import load_manifest
    scene = Path(args.scene or metadata["source_path"]).resolve()
    geometry = Path(args.geometry_model or metadata["geometry_model"]).resolve()
    radiometric = args.radiometric_dir or shared.get("radiometric_dir", "radiometric")
    load_manifest(scene, radiometric, shared["radiometric_protocol"])
    geometry_file = geometry / "point_cloud" / ("iteration_" + str(metadata["geometry_iteration"])) / "point_cloud.ply"
    digest = hashlib.sha256()
    with geometry_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata["geometry_sha256"]:
        raise ValueError("Geometry differs from the reference")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new directory")
    for protected in (scene, geometry, Path(args.reference).resolve().parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output must be separate from existing data/model directories")
    expected_split = metadata["camera_split"]
    command = [sys.executable, str(project / "train_thermal_physics.py"),
        "-s", str(scene), "-m", str(output), "--data_branch", "rgbt", "--eval",
        "--stage", "galpha", "--stage2_checkpoint", str(Path(args.reference).resolve()),
        "--geometry_model", str(geometry), "--geometry_iteration", str(metadata["geometry_iteration"]),
        "--radiometric_dir", radiometric, "--steps", str(args.steps), "--stage2_stop_mode", "budget",
        "--seed", str(metadata.get("seed", 0)), "--resolution", str(shared.get("resolution", -1)),
        "--ir_opacity_mode", args.mode, "--ir_opacity_logit_bound", str(args.opacity_bound),
        "--ir_opacity_lr", str(args.opacity_lr), "--ir_opacity_lr_final", str(args.opacity_lr_final),
        "--lambda_ir_opacity", str(args.lambda_opacity), "--lambda_tv", str(args.lambda_tv),
        "--temperature_lr", str(args.temperature_lr), "--temperature_lr_final", str(args.temperature_lr_final)]
    print("[G-alpha plan] " + json.dumps({"reference": args.reference, "output": str(output),
        "mode": args.mode, "steps": args.steps, "lambda_tv": args.lambda_tv,
        "opacity_logit_bound": args.opacity_bound, "RGB_geometry_frozen": True}), flush=True)
    del checkpoint
    # Mandatory CPU gradient/serialization and renderer-contract tests before a GPU run.
    subprocess.run([sys.executable, str(project / "tools/test_thermal_opacity.py"), "--require_torch"], check=True)
    subprocess.run(command, check=True)
    selected = output / "thermal_stage2_common.pt"
    trained = torch.load(selected, map_location="cpu")
    if trained["metadata"]["camera_split"] != expected_split:
        raise ValueError("Adaptation split differs from reference")
    display = output / "pseudocolor_calibration.json"
    subprocess.run([sys.executable, str(project / "tools/calibrate_thermal_pseudocolor.py"),
        "--scene", str(scene), "--checkpoint", str(selected), "--radiometric_dir", radiometric,
        "--output", str(display)], check=True)
    subprocess.run([sys.executable, str(project / "tools/evaluate_thermal_physics.py"),
        "-s", str(scene), "--geometry_model", str(geometry), "--radiometric_dir", radiometric,
        "--checkpoint", str(selected), "--thermal_output", "pseudocolor",
        "--display_calibration", str(display), "--split", "all", "--save_images",
        "--output_dir", str(output / "evaluation_pseudocolor")], check=True)


if __name__ == "__main__":
    main()
