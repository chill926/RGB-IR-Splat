"""Small validation scan for lambda_k and lambda_delta_epsilon.

Run this on the synthetic validation scene, then reuse the selected values for
all real/test scenes as required by the thesis protocol.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def floats(value):
    return [float(item) for item in value.split(",")]


def ints(value):
    return [int(item) for item in value.split(",")]


def final_validation_metrics(log_path):
    final_training = None
    final_validation = None
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "loss" in record:
                final_training = record
            if "validation_radiance_loss" in record:
                final_validation = record
    if final_training is None or final_validation is None:
        raise RuntimeError(f"No validation loss found in {log_path}")
    temperature_rmse = final_training.get("temperature_rmse_K")
    temperature_mae = final_training.get("temperature_mae_K")
    if temperature_rmse is None or temperature_mae is None:
        raise RuntimeError(
            "Synthetic regularization selection requires per-Gaussian temperature truth. "
            "Create stage 2 with --temperature_gt and keep it in the shared checkpoint protocol."
        )
    return {
        "temperature_rmse_K": float(temperature_rmse),
        "temperature_mae_K": float(temperature_mae),
        "validation_radiance_loss": float(final_validation["validation_radiance_loss"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--geometry_model", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--confirm_synthetic_validation", action="store_true", required=True,
                        help="Required acknowledgement that --source is synthetic validation, not a test scene")
    parser.add_argument("--lambda_k_grid", default="1e-5,1e-4,1e-3")
    parser.add_argument("--lambda_delta_grid", default="1e-5,1e-4,1e-3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--k_warmup_steps", type=int, default=500,
                        help="Nk fixed by this protocol for every later scene")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    grids = {"K": floats(args.lambda_k_grid), "R": floats(args.lambda_delta_grid)}
    for branch, grid in grids.items():
        option = "--lambda_k" if branch == "K" else "--lambda_delta_epsilon"
        for regularization in grid:
            radiance_losses = []
            temperature_rmse = []
            temperature_mae = []
            parameters = []
            for seed in ints(args.seeds):
                run_dir = output_root / f"{branch}_lambda_{regularization:g}_seed_{seed}"
                command = [sys.executable, "train_thermal_physics.py", "-s", args.source, "-m", str(run_dir),
                    "--eval",
                    "--stage", "branch", "--branch", branch, "--geometry_model", args.geometry_model,
                    "--stage2_checkpoint", args.stage2_checkpoint, "--steps", str(args.steps),
                    "--seed", str(seed), "--k_warmup_steps", str(args.k_warmup_steps),
                    option, str(regularization), "--regularization_scan_run"]
                subprocess.run(command, check=True)
                metrics = final_validation_metrics(run_dir / "thermal_training.jsonl")
                radiance_losses.append(metrics["validation_radiance_loss"])
                temperature_rmse.append(metrics["temperature_rmse_K"])
                temperature_mae.append(metrics["temperature_mae_K"])
                checkpoint = torch.load(run_dir / f"thermal_{branch}_final.pt", map_location="cpu")
                if branch == "K":
                    value = checkpoint["config"]["k_max"] * torch.tanh(checkpoint["state_dict"]["k_raw"])
                else:
                    value = checkpoint["config"]["delta_epsilon_max"] * torch.tanh(
                        checkpoint["state_dict"]["delta_epsilon_raw"])
                parameters.append([float(item) for item in value])
            results.append({"branch": branch, "regularization": regularization,
                "temperature_rmse_mean_K": statistics.mean(temperature_rmse),
                "temperature_rmse_std_K": statistics.pstdev(temperature_rmse),
                "temperature_mae_mean_K": statistics.mean(temperature_mae),
                "temperature_mae_std_K": statistics.pstdev(temperature_mae),
                "validation_mean": statistics.mean(radiance_losses),
                "validation_std": statistics.pstdev(radiance_losses), "seeds": ints(args.seeds),
                "parameter_nonmetal_mean": statistics.mean(row[0] for row in parameters),
                "parameter_nonmetal_std": statistics.pstdev(row[0] for row in parameters),
                "parameter_metal_mean": statistics.mean(row[1] for row in parameters),
                "parameter_metal_std": statistics.pstdev(row[1] for row in parameters)})
    selected = {}
    for branch in ("K", "R"):
        selected[branch] = min(
            (row for row in results if row["branch"] == branch),
            key=lambda row: (row["temperature_rmse_mean_K"], row["validation_mean"]),
        )
    report = {"selection_dataset_role": "synthetic_validation",
              "source": os.path.abspath(args.source),
              "k_warmup_steps": args.k_warmup_steps,
              "selection_metric": "final_temperature_rmse_K_then_validation_radiance_loss",
              "results": results, "selected": selected,
              "instruction": "Freeze these selected values for every subsequent test scene."}
    with open(output_root / "regularization_scan.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
