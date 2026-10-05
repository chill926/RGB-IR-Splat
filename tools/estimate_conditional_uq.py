"""Estimate conditional parameter uncertainty from repeated C/K/R runs.

The reported spread is conditional on frozen geometry/cameras, material masks,
fixed epsilon_0 values, observation conversion and the implemented forward
model. It is not total physical uncertainty.
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError("Conditional UQ needs at least two independently seeded checkpoints")
    loaded = [torch.load(path, map_location="cpu") for path in args.checkpoints]
    branches = {checkpoint.get("branch") for checkpoint in loaded}
    geometry = {checkpoint.get("metadata", {}).get("geometry_sha256") for checkpoint in loaded}
    material_names = {tuple(checkpoint["config"].get("material_names", [])) for checkpoint in loaded}
    if len(branches) != 1 or len(geometry) != 1 or len(material_names) != 1:
        raise ValueError("UQ checkpoints must share branch, frozen geometry and material definitions")
    temperature = torch.stack([
        checkpoint["config"]["temp_min"] +
        (checkpoint["config"]["temp_max"] - checkpoint["config"]["temp_min"]) *
        torch.sigmoid(checkpoint["state_dict"]["temperature_raw"])
        for checkpoint in loaded
    ])
    environment = torch.stack([
        torch.sigmoid(checkpoint["state_dict"]["environment_raw"]).reshape(()) for checkpoint in loaded
    ])
    branch = next(iter(branches))
    result = {
        "format_version": 1,
        "branch": branch,
        "runs": len(loaded),
        "conditioning": (
            "Frozen geometry/cameras, fixed material labels and epsilon_0, observation conversion, "
            "regularization and forward-model assumptions."
        ),
        "temperature_mean_K": temperature.mean(dim=0),
        "temperature_std_K": temperature.std(dim=0, unbiased=False),
        "environment_mean": environment.mean(),
        "environment_std": environment.std(unbiased=False),
        "material_names": list(next(iter(material_names))),
        "checkpoint_paths": [str(Path(path).resolve()) for path in args.checkpoints],
    }
    if branch == "K":
        values = torch.stack([checkpoint["config"]["k_max"] *
                              torch.tanh(checkpoint["state_dict"]["k_raw"])
                              for checkpoint in loaded])
        result["k_mean_by_material"] = values.mean(dim=0)
        result["k_std_by_material"] = values.std(dim=0, unbiased=False)
    elif branch == "R":
        values = torch.stack([checkpoint["config"]["delta_epsilon_max"] *
                              torch.tanh(checkpoint["state_dict"]["delta_epsilon_raw"])
                              for checkpoint in loaded])
        result["delta_epsilon_mean_by_material"] = values.mean(dim=0)
        result["delta_epsilon_std_by_material"] = values.std(dim=0, unbiased=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    summary = {
        "branch": branch, "runs": len(loaded),
        "temperature_std_mean_K": float(result["temperature_std_K"].mean()),
        "environment_std": float(result["environment_std"]),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
