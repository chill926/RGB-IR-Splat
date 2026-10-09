"""Preflight, fresh experiment D, then RGB/IR evaluation using trained geometry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def build_joint_command(args, metadata, config, project, scene, geometry, masks, materials):
    from tools.run_sh_residual import build_training_command
    command = build_training_command(args, metadata, config, project, scene, geometry, masks, materials)
    command[1] = str(project / "train_thermal_joint.py")
    command.extend(["--joint_reference", str(Path(args.reference).resolve()),
        "--display_calibration", str(Path(args.display_calibration).resolve()),
        "--joint_start_step", str(args.joint_start_step), "--eval_every", "500", "--save_every", "5000"])
    for name in ("lambda_rgb", "lambda_signal", "lambda_display", "joint_position_lr", "joint_position_lr_final",
                 "joint_scaling_lr", "joint_rotation_lr", "joint_opacity_lr", "joint_rgb_lr"):
        command.extend(["--" + name, str(getattr(args, name))])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Coordinate-repaired SH2 Stage2 checkpoint: settings only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--display_calibration", required=True)
    for name in ("scene", "geometry_model", "material_mask_dir", "material_config", "radiometric_dir"):
        parser.add_argument("--" + name, default="")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--joint_start_step", type=int, default=3000)
    parser.add_argument("--start_step", type=int, default=3000, help="SH activation step")
    parser.add_argument("--bound", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--lr_final", type=float, default=.0001)
    parser.add_argument("--regularization", type=float, default=.001)
    for name, value in (("lambda_rgb", 1.), ("lambda_signal", 1.), ("lambda_display", 1.),
            ("joint_position_lr", 1.6e-5), ("joint_position_lr_final", 1.6e-7),
            ("joint_scaling_lr", 5e-4), ("joint_rotation_lr", 1e-4), ("joint_opacity_lr", 5e-3), ("joint_rgb_lr", 2.5e-4)):
        parser.add_argument("--" + name, type=float, default=value)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    import torch
    from utils.thermal_pseudocolor import FrozenDisplay, load_manifest
    from utils.flir_radiometry import file_sha256
    checkpoint = torch.load(args.reference, map_location="cpu")
    metadata, config = checkpoint["metadata"], checkpoint["config"]
    shared = metadata["shared_training_protocol"]
    if (checkpoint.get("branch") != "stage2" or checkpoint.get("joint_geometry") is not None
            or shared.get("observation_domain") != "raw_rjpeg" or shared["radiometric_protocol"].get("format_version") != 2):
        raise ValueError("Use the coordinate-repaired frozen Stage2 checkpoint as D reference")
    if shared.get("oracle_temperature_supervision") or shared.get("lambda_temperature_supervision", 0):
        raise ValueError("D does not use oracle temperature supervision")
    scene = Path(args.scene or metadata["source_path"]).resolve()
    geometry = Path(args.geometry_model or metadata["geometry_model"]).resolve()
    materials = Path(args.material_config or metadata["material_config"]).resolve()
    masks = Path(args.material_mask_dir or metadata["material_mask_dir"]).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("D output already contains files; choose a new experiment directory")
    for protected in (scene, geometry, Path(args.reference).resolve().parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("D output overlaps existing data/models")
    args.radiometric_dir = args.radiometric_dir or shared["radiometric_dir"]
    _, manifest, _ = load_manifest(scene, args.radiometric_dir, shared["radiometric_protocol"])
    if manifest["coordinate_alignment"]["camera_split"] != metadata["camera_split"]:
        raise ValueError("D coordinate split changed")
    FrozenDisplay(args.display_calibration, shared["radiometric_protocol"], metadata["camera_split"]["fit"])
    if file_sha256(materials) != metadata["material_config_sha256"]:
        raise ValueError("D material configuration changed")
    geometry_file = geometry / "point_cloud" / ("iteration_" + str(metadata["geometry_iteration"])) / "point_cloud.ply"
    if file_sha256(geometry_file) != metadata["geometry_sha256"]:
        raise ValueError("D RGB initializer changed")
    args.degree, args.lambda_tv = 2, 0.0
    command = build_joint_command(args, metadata, config, project, scene, geometry, masks, materials)
    plan = {"experiment": "D", "reference_settings_only": str(Path(args.reference).resolve()),
        "Stage2_weights_loaded": False, "fixed_gaussian_count": True, "command": command}
    print("[D plan] " + json.dumps(plan), flush=True)
    del checkpoint
    for test in ("test_thermal_coordinates.py", "test_thermal_sh_residual.py", "test_thermal_joint.py"):
        subprocess.run([sys.executable, str(project / "tools" / test), "--require_torch"], check=True)
    subprocess.run(command, check=True)
    selected = output / "thermal_joint_D_selected.pt"
    result = torch.load(selected, map_location="cpu")
    if result["metadata"]["camera_split"] != metadata["camera_split"] or "joint_geometry" not in result:
        raise ValueError("D selected checkpoint provenance differs")
    (output / "joint_experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    del result
    common = [sys.executable, str(project / "tools/evaluate_thermal_physics.py"), "-s", str(scene),
        "--geometry_model", str(geometry), "--radiometric_dir", args.radiometric_dir,
        "--checkpoint", str(selected), "--split", "all", "--save_images"]
    subprocess.run(common + ["--thermal_output", "pseudocolor", "--display_calibration", args.display_calibration,
        "--output_dir", str(output / "evaluation_pseudocolor")], check=True)
    subprocess.run(common + ["--modality", "rgb", "--output_dir", str(output / "evaluation_rgb")], check=True)
    initial_rgb = [sys.executable, str(project / "tools/evaluate_thermal_physics.py"), "-s", str(scene),
        "--geometry_model", str(geometry), "--checkpoint", str(Path(args.reference).resolve()),
        "--modality", "rgb", "--split", "all", "--output_dir", str(output / "evaluation_rgb_initializer")]
    subprocess.run(initial_rgb, check=True)
    comparison = {"baseline_reference": str(Path(args.reference).resolve()), "splits": {}}
    rgb = json.loads((output / "evaluation_rgb/metrics.json").read_text())
    initial = json.loads((output / "evaluation_rgb_initializer/metrics.json").read_text())
    thermal = json.loads((output / "evaluation_pseudocolor/metrics.json").read_text())
    baseline_path = Path(args.reference).resolve().parent / "evaluation_pseudocolor/metrics.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.is_file() else None
    if baseline is not None and (baseline.get("radiometric_protocol") != thermal.get("radiometric_protocol") or
            baseline.get("display_calibration_sha256") != thermal.get("display_calibration_sha256")):
        raise ValueError("Baseline evaluation uses a different input/display protocol; do not compare")
    for split in ("fit", "validation", "test"):
        row = {"D_thermal_PSNR": thermal["splits"][split]["PSNR"],
               "D_RGB_PSNR": rgb["splits"][split]["PSNR"],
               "RGB_initializer_PSNR": initial["splits"][split]["PSNR"],
               "RGB_PSNR_delta": rgb["splits"][split]["PSNR"] - initial["splits"][split]["PSNR"]}
        if baseline is not None:
            row.update(baseline_thermal_PSNR=baseline["splits"][split]["PSNR"],
                thermal_PSNR_delta=row["D_thermal_PSNR"] - baseline["splits"][split]["PSNR"])
        comparison["splits"][split] = row
    (output / "joint_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("[D comparison] " + json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
