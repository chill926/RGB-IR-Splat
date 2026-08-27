"""Validate and summarize the equal-budget C/K/R comparison."""
import argparse
import json
from pathlib import Path

import torch


def last_records(path):
    training = evaluation = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "loss" in record:
                training = record
            if "validation_radiance_loss" in record:
                evaluation = record
    if training is None or evaluation is None:
        raise RuntimeError(f"Incomplete branch log: {path}")
    return training, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    rows = {}
    protocol_hash = None
    common_step = common_seed = None
    for branch in ("C", "K", "R"):
        branch_root = root / f"branch_{branch}"
        checkpoint = torch.load(branch_root / f"thermal_{branch}_final.pt", map_location="cpu")
        metadata = checkpoint.get("metadata", {})
        current_hash = metadata.get("comparison_protocol_sha256")
        if not current_hash or (protocol_hash is not None and current_hash != protocol_hash):
            raise RuntimeError("C/K/R do not share one comparison protocol")
        protocol_hash = current_hash
        step, seed = int(checkpoint["step"]), int(metadata["seed"])
        if common_step is not None and (step != common_step or seed != common_seed):
            raise RuntimeError("C/K/R final update counts or random seeds differ")
        common_step, common_seed = step, seed
        training, evaluation = last_records(branch_root / "thermal_training.jsonl")
        rows[branch] = {
            "updates": step, "seed": seed,
            "training_loss": training["loss"],
            "training_radiance_loss": training["radiance_loss"],
            "validation_radiance_loss": evaluation["validation_radiance_loss"],
            "temperature_mae_K": training.get("temperature_mae_K"),
            "temperature_rmse_K": training.get("temperature_rmse_K"),
            "temperature_semantics": training.get("temperature_semantics"),
            "parameters_stable": evaluation.get("parameters_stable"),
            "material_parameter_update": evaluation.get("material_parameter_update"),
            "k_nonmetal": training.get("k_nonmetal"), "k_metal": training.get("k_metal"),
            "delta_epsilon_nonmetal": training.get("delta_epsilon_nonmetal"),
            "delta_epsilon_metal": training.get("delta_epsilon_metal"),
        }
    report = {"comparison_protocol_sha256": protocol_hash, "branches": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
