"""Fresh Stage2 SH2 experiment (or matched no-SH control), then pseudocolor evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def build_training_command(args, metadata, config, project, scene, geometry, masks, materials):
    """The reference provides settings only. Never pass Stage2 weights to training."""
    shared = metadata["shared_training_protocol"]
    command = [sys.executable, str(project / "train_thermal_physics.py"),
        "-s", str(scene), "-m", str(Path(args.output).resolve()),
        "--data_branch", "rgbt", "--eval", "--stage", "stage2",
        "--geometry_model", str(geometry), "--geometry_iteration", str(metadata["geometry_iteration"]),
        "--material_mask_dir", str(masks), "--material_config", str(materials),
        "--observation_domain", "raw_rjpeg", "--radiometric_dir", args.radiometric_dir or shared.get("radiometric_dir", "radiometric"),
        "--steps", str(args.steps), "--stage2_stop_mode", "budget",
        "--seed", str(metadata.get("seed", 0)), "--resolution", str(shared.get("resolution", -1)),
        "--temp_min", str(config["temp_min"]), "--temp_max", str(config["temp_max"]), "--temp_ref", str(config["temp_ref"]),
        "--sh_residual_degree", str(args.degree), "--sh_residual_start_step", str(args.start_step),
        "--sh_residual_bound", str(args.bound), "--sh_residual_lr", str(args.lr),
        "--sh_residual_lr_final", str(args.lr_final), "--lambda_sh_residual", str(args.regularization)]
    for name in ("temperature_lr", "temperature_lr_final", "environment_lr", "environment_lr_final",
                 "lambda_tv", "lambda_environment", "environment_prior_beta", "environment_init_quantile",
                 "huber_delta", "tv_neighbors", "validation_fraction", "validation_seed",
                 "loss_scale_mode", "loss_scale_floor", "tv_temperature_scale_K",
                 "tv_material_boundary_weight", "lr_hold_fraction"):
        if name in shared:
            command.extend(["--" + name, str(shared[name])])
    if args.lambda_tv is not None:
        command.extend(["--lambda_tv", str(args.lambda_tv)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Existing raw_rjpeg Stage2 checkpoint: settings only, no weights")
    parser.add_argument("--output", required=True, help="New, empty experiment directory")
    parser.add_argument("--degree", type=int, choices=(0, 2), default=2, help="Use 0 for a matched fresh physical baseline")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--start_step", type=int, default=3000)
    parser.add_argument("--bound", type=float, default=0.5, help="Residual cap / fit loss scale")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_final", type=float, default=0.0001)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--lambda_tv", type=float, default=None, help="Optional shared override for both matched runs")
    parser.add_argument("--scene", default="")
    parser.add_argument("--geometry_model", default="")
    parser.add_argument("--material_config", default="")
    parser.add_argument("--material_mask_dir", default="")
    parser.add_argument("--radiometric_dir", default="")
    parser.add_argument("--display_calibration", default="", help="Reuse the same fit-only mapping for baseline and SH")
    parser.add_argument("--allow_new_radiometric", action="store_true",
                        help="Fresh fit only: accept new aligned data with identical camera response and exact recorded split")
    args = parser.parse_args()
    if args.steps <= 0 or (args.degree and not 1 <= args.start_step <= args.steps):
        raise ValueError("Positive budget and SH start step inside budget are required")
    if (not all(math.isfinite(x) for x in (args.bound, args.lr, args.lr_final, args.regularization))
            or args.bound <= 0 or not 0 < args.lr_final <= args.lr or args.regularization < 0
            or (args.lambda_tv is not None and (not math.isfinite(args.lambda_tv) or args.lambda_tv < 0))):
        raise ValueError("Invalid SH/TV hyperparameters")
    import torch
    checkpoint = torch.load(args.reference, map_location="cpu")
    metadata, config = checkpoint.get("metadata", {}), checkpoint["config"]
    shared = metadata.get("shared_training_protocol", {})
    if checkpoint.get("branch") != "stage2" or shared.get("observation_domain") != "raw_rjpeg":
        raise ValueError("Reference must be a raw_rjpeg Stage2 checkpoint")
    if shared.get("oracle_temperature_supervision") or shared.get("lambda_temperature_supervision", 0):
        raise ValueError("This experiment runner does not use Oracle temperature supervision")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from utils.thermal_pseudocolor import load_manifest, FrozenDisplay
    reference = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    scene = Path(args.scene or metadata["source_path"]).resolve()
    geometry = Path(args.geometry_model or metadata["geometry_model"]).resolve()
    materials = Path(args.material_config or metadata.get("material_config", "")).resolve()
    masks = Path(args.material_mask_dir or metadata.get("material_mask_dir", materials.parent)).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new directory")
    for protected in (scene, geometry, reference.parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output must be separate from existing data/model directories")
    if not materials.is_file():
        raise ValueError("Material config unavailable; pass --material_config")
    if metadata.get("material_config_sha256") and hashlib.sha256(materials.read_bytes()).hexdigest() != metadata["material_config_sha256"]:
        raise ValueError("Material config differs from the reference")
    names = metadata.get("camera_split", {}).get("fit", [])
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if not names or any(not any((folder / (name + extension)).is_file()
                         for folder in (masks, masks / "train", masks / "test") for extension in extensions)
                        for name in names):
        raise ValueError("Reference fit masks unavailable; pass --material_mask_dir")
    radiometric = args.radiometric_dir or shared.get("radiometric_dir", "radiometric")
    _, manifest, manifest_digest = load_manifest(scene, radiometric,
        None if args.allow_new_radiometric else shared.get("radiometric_protocol"))
    if args.allow_new_radiometric:
        if manifest["signal_calibration"] != shared["radiometric_protocol"]["signal_calibration"]:
            raise ValueError("New radiometric data changes camera response; this runner only accepts coordinate changes")
        alignment = manifest.get("coordinate_alignment")
        if alignment is None or alignment["camera_split"] != metadata["camera_split"]:
            raise ValueError("New aligned data must use the reference's exact fit/validation/test split")
    geometry_file = geometry / "point_cloud" / ("iteration_" + str(metadata["geometry_iteration"])) / "point_cloud.ply"
    digest = hashlib.sha256()
    with geometry_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata.get("geometry_sha256"):
        raise ValueError("Stage1 geometry differs from the reference")
    display = Path(args.display_calibration).resolve() if args.display_calibration else (
        output / "pseudocolor_calibration.json" if args.allow_new_radiometric
        else reference.parent / "pseudocolor_calibration.json")
    if args.display_calibration and not display.is_file():
        raise ValueError("Requested display calibration does not exist")
    if display.is_file():
        expected = {"manifest_sha256": manifest_digest, "signal_calibration": manifest["signal_calibration"]}
        FrozenDisplay(display, expected, names)
    command = build_training_command(args, metadata, config, project, scene, geometry, masks, materials)
    plan = {"reference_settings_only": str(reference), "fresh_stage2_fit": True,
        "Stage2_weights_loaded": False, "IR_opacity_weights_loaded": False,
        "degree": args.degree, "steps": args.steps, "start_step": args.start_step,
        "bound_fraction_of_loss_scale": args.bound, "output": str(output), "command": command}
    plan.update(allow_new_radiometric=args.allow_new_radiometric, radiometric_manifest_sha256=manifest_digest)
    print("[SH Stage2 plan] " + json.dumps(plan), flush=True)
    del checkpoint
    subprocess.run([sys.executable, str(project / "tools/test_thermal_coordinates.py"), "--require_torch"], check=True)
    subprocess.run([sys.executable, str(project / "tools/test_thermal_sh_residual.py"), "--require_torch"], check=True)
    subprocess.run(command, check=True)
    selected = output / "thermal_stage2_common.pt"
    trained = torch.load(selected, map_location="cpu")
    if trained["metadata"]["camera_split"] != metadata["camera_split"]:
        raise ValueError("Fresh Stage2 split differs from reference; do not compare results")
    if trained["metadata"].get("warm_start_checkpoint_sha256") is not None:
        raise ValueError("Expected a fresh Stage2 model")
    if trained["metadata"]["shared_training_protocol"]["radiometric_protocol"]["manifest_sha256"] != manifest_digest:
        raise ValueError("Training did not use the planned radiometric input")
    (output / "sh_experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if not display.is_file():
        display = output / "pseudocolor_calibration.json"
        subprocess.run([sys.executable, str(project / "tools/calibrate_thermal_pseudocolor.py"),
            "--scene", str(scene), "--radiometric_dir", radiometric, "--checkpoint", str(selected),
            "--output", str(display)], check=True)
    evaluation = [sys.executable, str(project / "tools/evaluate_thermal_physics.py"),
        "-s", str(scene), "--geometry_model", str(geometry), "--radiometric_dir", radiometric,
        "--checkpoint", str(selected), "--thermal_output", "pseudocolor", "--display_calibration", str(display),
        "--split", "all", "--save_images"]
    subprocess.run(evaluation + ["--output_dir", str(output / "evaluation_pseudocolor")], check=True)
    if args.degree == 2:
        subprocess.run(evaluation + ["--physical_only", "--output_dir", str(output / "evaluation_physical_only")], check=True)


if __name__ == "__main__":
    main()
