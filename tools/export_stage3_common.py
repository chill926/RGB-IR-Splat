"""Export the complete trained D state for frozen-geometry C/R/K pilots."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.thermal_stage3 import atomic_save, export_common, file_sha256, validate_common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learn_materials", required=True,
                        help="Explicit comma-separated names, or preserve (requires matched D flags)")
    parser.add_argument("--reuse_if_matching", action="store_true",
                        help="Reuse an immutable common only when D hash and candidates match")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    original = torch.load(args.checkpoint, map_location="cpu")
    if args.learn_materials == "preserve":
        state = original["state_dict"]
        if not torch.equal(state["learn_k_by_material"], state["learn_delta_by_material"]):
            raise ValueError("D flags differ; supply explicit candidate names")
        names = [name for name, active in zip(original["config"]["material_names"],
                                             state["learn_k_by_material"].tolist()) if active]
    else:
        names = [name.strip() for name in args.learn_materials.split(",") if name.strip()]
    if output.exists():
        if not args.reuse_if_matching:
            raise ValueError("Common output already exists; never overwrite an experiment")
        report = validate_common(torch.load(output, map_location="cpu"))
        if (report["source_checkpoint_sha256"] != file_sha256(args.checkpoint)
                or report["enabled_materials"] != names):
            raise ValueError("Existing common has different D input/material candidates")
        print(json.dumps({"common": str(output), "reused": True, "material_policy": report["material_policy"]}), flush=True)
        return
    payload = export_common(original, args.checkpoint, names)
    atomic_save(payload, output)
    report = payload["metadata"]["stage3_common"]
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"common": str(output), "material_policy": report["material_policy"],
                      "unknown_gaussians": report["unknown_gaussians"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
