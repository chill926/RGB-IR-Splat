"""Retry Stage2 from recorded data/geometry settings, then evaluate pseudocolor.

Run on the physir CUDA server. The reference supplies configuration, not weights:
this is a fresh Stage2 fit with fixed loss units and no temperature TV by default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Existing raw_rjpeg Stage2 checkpoint")
    parser.add_argument("--output", required=True, help="New, empty output directory")
    parser.add_argument("--scene", default="")
    parser.add_argument("--geometry_model", default="")
    parser.add_argument("--material_config", default="")
    parser.add_argument("--material_mask_dir", default="")
    parser.add_argument("--radiometric_dir", default="")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lambda_tv", type=float, default=0.0)
    parser.add_argument("--quality_apparent_mae_K", type=float, default=None)
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(args.reference, map_location="cpu")
    metadata, config = checkpoint.get("metadata", {}), checkpoint["config"]
    shared = metadata.get("shared_training_protocol", {})
    if checkpoint.get("branch") != "stage2" or shared.get("observation_domain") != "raw_rjpeg":
        raise ValueError("Reference must be an original-signal raw_rjpeg Stage2 checkpoint")
    project = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new directory")
    scene = Path(args.scene or metadata["source_path"]).resolve()
    geometry = Path(args.geometry_model or metadata["geometry_model"]).resolve()
    material_config = Path(args.material_config or metadata.get("material_config", "")).resolve()
    if not material_config.is_file():
        raise ValueError("Recorded material config is unavailable; pass --material_config")
    masks = Path(args.material_mask_dir or metadata.get("material_mask_dir", material_config.parent)).resolve()
    names = metadata.get("camera_split", {}).get("fit", [])
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if not names or any(not any((folder / (name + extension)).is_file()
                         for folder in (masks, masks / "train", masks / "test") for extension in extensions)
                        for name in names):
        raise ValueError("Cannot locate all reference fit masks; pass the original --material_mask_dir")
    if args.steps <= 0 or args.lambda_tv < 0:
        raise ValueError("Steps must be positive and TV weight non-negative")
    for protected in (scene, geometry, Path(args.reference).resolve().parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output must be separate from existing data/model directories")

    sys.path.insert(0, str(project))
    from utils.thermal_pseudocolor import load_manifest
    load_manifest(scene, args.radiometric_dir or shared.get("radiometric_dir", "radiometric"),
                  shared.get("radiometric_protocol"))
    geometry_file = geometry / "point_cloud" / ("iteration_" + str(metadata["geometry_iteration"])) / "point_cloud.ply"
    with geometry_file.open("rb") as handle:
        geometry_digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            geometry_digest.update(block)
    if geometry_digest.hexdigest() != metadata.get("geometry_sha256"):
        raise ValueError("Geometry content differs from reference checkpoint")
    if metadata.get("material_config_sha256"):
        digest = hashlib.sha256(material_config.read_bytes()).hexdigest()
        if digest != metadata["material_config_sha256"]:
            raise ValueError("Material config content differs from reference checkpoint")
    del checkpoint
    command = [sys.executable, str(project / "train_thermal_physics.py"),
        "-s", str(scene), "-m", str(output), "--data_branch", "rgbt", "--eval", "--stage", "stage2",
        "--geometry_model", str(geometry), "--geometry_iteration", str(metadata["geometry_iteration"]),
        "--material_mask_dir", str(masks), "--material_config", str(material_config),
        "--observation_domain", "raw_rjpeg", "--radiometric_dir", args.radiometric_dir or shared.get("radiometric_dir", "radiometric"),
        "--temp_min", str(config["temp_min"]), "--temp_max", str(config["temp_max"]),
        "--temp_ref", str(config["temp_ref"]), "--loss_scale_mode", "fit_quantile",
        "--huber_delta", "0.02", "--lambda_tv", str(args.lambda_tv), "--tv_temperature_scale_K", "10",
        "--lr_hold_fraction", "0.5", "--stage2_stop_mode", "budget", "--steps", str(args.steps),
        "--validation_fraction", str(shared.get("validation_fraction", 0.1)),
        "--validation_seed", str(shared.get("validation_seed", 2027)),
        "--seed", str(metadata.get("seed", 0)), "--resolution", str(shared.get("resolution", -1))]
    # Keep all reference fit settings except the explicitly changed training policy.
    for name in ("temperature_lr", "temperature_lr_final", "environment_lr", "environment_lr_final",
                 "lambda_environment", "environment_prior_beta", "environment_init_quantile"):
        if name in shared:
            command.extend(["--" + name, str(shared[name])])
    if args.quality_apparent_mae_K is not None:
        command.extend(["--quality_apparent_mae_K", str(args.quality_apparent_mae_K)])
    print("[retry-plan] " + json.dumps({"reference": str(Path(args.reference).resolve()),
        "output": str(output), "geometry": str(geometry), "material_masks": str(masks),
        "steps": args.steps, "lambda_tv": args.lambda_tv, "fresh_stage2_fit": True}), flush=True)
    subprocess.run([sys.executable, str(project / "tools/test_thermal_training.py")], check=True)
    subprocess.run(command, check=True)
    selected = output / "thermal_stage2_common.pt"
    trained = torch.load(selected, map_location="cpu")
    if trained["metadata"]["camera_split"] != metadata["camera_split"]:
        raise ValueError("Retry camera split differs from reference; do not compare results")
    display = output / "pseudocolor_calibration.json"
    subprocess.run([sys.executable, str(project / "tools/calibrate_thermal_pseudocolor.py"),
        "--scene", str(scene), "--checkpoint", str(selected), "--output", str(display)], check=True)
    subprocess.run([sys.executable, str(project / "tools/evaluate_thermal_physics.py"),
        "-s", str(scene), "--checkpoint", str(selected), "--thermal_output", "pseudocolor",
        "--display_calibration", str(display), "--split", "all", "--save_images",
        "--output_dir", str(output / "evaluation_pseudocolor")], check=True)


if __name__ == "__main__":
    main()
