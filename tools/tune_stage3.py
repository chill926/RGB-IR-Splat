"""Validation-only Stage3 search: six C LR settings, then R/K regularization."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from utils.thermal_stage3 import file_sha256, verify_inputs


def rank_trial(trial):
    summary = trial["summary"]
    # If all best checkpoints are step0, prefer the least degraded final
    # validation objective. This does not claim a trained improvement.
    return (summary["best_validation_objective"],
            summary["final_validation_metrics"]["selection_loss"], trial["id"])


def make_command(args, identifier, branch, t_lr, e_lr, regularization, lock):
    return [sys.executable, "-B", "-u", str(ROOT / "train_thermal_stage3.py"),
            "--common", str(args.common), "--branch", branch,
            "--comparison_protocol", str(lock), "-m", str(args.output / identifier),
            "--steps", str(args.steps), "--seed", str(args.seed),
            "--eval_every", str(args.eval_every), "--save_every", str(args.save_every),
            "--log_every", "100", "--temperature_lr", str(t_lr),
            "--temperature_lr_final", str(t_lr * .01), "--environment_lr", str(e_lr),
            "--environment_lr_final", str(e_lr * .01),
            "--material_lr", str(args.material_lr), "--material_lr_final", str(args.material_lr_final),
            "--lr_hold_fraction", str(args.lr_hold_fraction),
            "--lambda_k", str(regularization), "--lambda_delta_epsilon", str(regularization)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, default=ROOT / "output/RoadBlock_stage3_D_pilot_run1_common/thermal_stage3_common.pt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--material_lr", type=float, default=1e-4)
    parser.add_argument("--material_lr_final", type=float, default=1e-6)
    parser.add_argument("--lr_hold_fraction", type=float, default=.1)
    parser.add_argument("--resume", action="store_true", help="Skip matching completed trials; never overwrite incomplete ones")
    parser.add_argument("--dry_run", action="store_true", help="Verify inputs and show search plan without creating outputs/training")
    args = parser.parse_args()
    args.common, args.output = args.common.resolve(), args.output.resolve()
    if min(args.steps, args.eval_every, args.save_every) <= 0:
        parser.error("Budget and intervals must be positive")
    if (not all(math.isfinite(v) for v in (args.material_lr, args.material_lr_final, args.lr_hold_fraction))
            or not 0 < args.material_lr_final <= args.material_lr or not 0 <= args.lr_hold_fraction < 1):
        parser.error("Invalid material LR/hold fraction")
    return args


def main():
    args = parse_args()
    checkpoint = torch.load(args.common, map_location="cpu")
    verify_inputs(checkpoint)
    metadata = checkpoint["metadata"]
    for protected in (args.common.parent, Path(metadata["source_path"]).resolve(),
                      Path(metadata["geometry_model"]).resolve(),
                      Path(metadata["stage3_common"]["source_checkpoint"]).resolve().parent):
        if args.output == protected or args.output in protected.parents or protected in args.output.parents:
            raise ValueError("Search output overlaps existing common/data/models")
    common = metadata["stage3_common"]
    if not common["enabled_materials"]:
        raise ValueError("R/K search needs explicitly enabled materials")
    c_grid = [(t, e) for t in (3e-5, 1e-4, 3e-4) for e in (1e-5, 1e-4)]
    regularization_grid = [1e-3, 1e-4, 1e-5]
    plan = {"kind": "stage3_validation_hyperparameter_search_v1",
            "common": str(args.common), "common_sha256": file_sha256(args.common),
            "seed": args.seed, "steps_per_trial": args.steps, "eval_every": args.eval_every,
            "save_every": args.save_every, "material_lr": args.material_lr,
            "material_lr_final": args.material_lr_final, "lr_hold_fraction": args.lr_hold_fraction,
            "C_temperature_environment_grid": [list(pair) for pair in c_grid],
            "R_K_regularization_grid": regularization_grid, "trial_count": 12,
            "selection": "best_validation_objective_then_final_validation_objective_then_trial_id",
            "test_evaluation_during_search": False, "enabled_materials": common["enabled_materials"],
            "runtime_source_sha256": {str(p.relative_to(ROOT)): file_sha256(p)
                for p in [Path(__file__).resolve(), ROOT / "train_thermal_stage3.py", ROOT / "utils/thermal_stage3.py",
                          ROOT / "utils/thermal_physics.py", ROOT / "utils/thermal_joint.py",
                          ROOT / "utils/thermal_sh_residual.py", ROOT / "utils/thermal_observations.py",
                          ROOT / "utils/thermal_training_utils.py", ROOT / "scene/__init__.py",
                          ROOT / "scene/dataset_readers.py", ROOT / "gaussian_renderer/__init__.py"]}}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists():
        if not args.resume or not (args.output / "search_plan.json").is_file():
            raise ValueError("Search output exists; choose a new output or --resume a matching search")
        if json.loads((args.output / "search_plan.json").read_text()) != plan:
            raise ValueError("Resume settings/source differ from the recorded search")
    else:
        args.output.mkdir(parents=True)
        (args.output / "search_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    trials = []
    def execute(identifier, branch, t_lr, e_lr, regularization, lock):
        output = args.output / identifier
        summary_path = output / "training_summary.json"
        command = make_command(args, identifier, branch, t_lr, e_lr, regularization, lock)
        settings = {"id": identifier, "branch": branch, "temperature_lr": t_lr,
                    "environment_lr": e_lr, "regularization": regularization, "command": command}
        print("[tune start] " + json.dumps(settings), flush=True)
        if summary_path.is_file() and args.resume:
            summary = json.loads(summary_path.read_text())
        else:
            if output.exists():
                raise ValueError("Incomplete trial exists; preserve it and use a new search output: " + str(output))
            with (args.output / (identifier + ".log")).open("x") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            summary = json.loads(summary_path.read_text())
        protocol = summary["protocol"]
        shared = protocol["shared_training_protocol"]
        expected = {"common_sha256": plan["common_sha256"], "seed": args.seed, "steps": args.steps,
                    "lambda_k": regularization, "lambda_delta_epsilon": regularization,
                    "material_lr": args.material_lr, "material_lr_final": args.material_lr_final,
                    "eval_every": args.eval_every, "save_every": args.save_every, "log_every": 100}
        if (any(protocol.get(key) != value for key, value in expected.items())
                or shared["temperature_lr"] != t_lr or shared["temperature_lr_final"] != t_lr * .01
                or shared["environment_lr"] != e_lr or shared["environment_lr_final"] != e_lr * .01
                or shared["lr_hold_fraction"] != args.lr_hold_fraction or summary["branch"] != branch
                or summary["stopped_step"] != args.steps or not summary["frozen_state_verified"]):
            raise ValueError("Trial protocol/completion mismatch: " + identifier)
        trial = dict(settings, summary=summary)
        trials.append(trial)
        (args.output / "trial_results.json").write_text(json.dumps(trials, indent=2, allow_nan=False) + "\n")
        print("[tune done] " + json.dumps({"id": identifier, "best_step": summary["best_step"],
              "best_validation": summary["best_validation_objective"],
              "final_validation": summary["final_validation_metrics"]["selection_loss"]}), flush=True)
        return trial
    candidates = []
    for index, (t_lr, e_lr) in enumerate(c_grid, 1):
        identifier = "C_lr_{:02d}".format(index)
        candidates.append(execute(identifier, "C", t_lr, e_lr, 1e-4,
                                  args.output / "locks" / (identifier + ".json")))
    selected_c = min(candidates, key=rank_trial)
    materials = {branch: [] for branch in ("R", "K")}
    for index, regularization in enumerate(regularization_grid, 1):
        lock = args.output / "locks" / ("material_reg_{:02d}.json".format(index))
        for branch in ("R", "K"):
            identifier = branch + "_reg_{:02d}".format(index)
            materials[branch].append(execute(identifier, branch, selected_c["temperature_lr"],
                                             selected_c["environment_lr"], regularization, lock))
    selected = {"C": selected_c, **{branch: min(rows, key=rank_trial) for branch, rows in materials.items()}}
    selection = {"selected_trial_ids": {b: p["id"] for b, p in selected.items()},
                 "selected_best_steps": {b: p["summary"]["best_step"] for b, p in selected.items()},
                 "selected_best_validation": {b: p["summary"]["best_validation_objective"] for b, p in selected.items()},
                 "C_all_best_steps_zero": all(p["summary"]["best_step"] == 0 for p in candidates),
                 "selection_uses_validation_only": True, "test_used": False,
                 "formal_run_needed_for_final_matched_C_R_K_comparison": True}
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    settings = {"GS_WORK": str(ROOT), "THERMAL_PYTHON": sys.executable,
                "STAGE3_COMMON": str(args.common), "STAGE3_D_CHECKPOINT": common["source_checkpoint"],
                "STAGE3_LEARN_MATERIALS": ",".join(common["enabled_materials"]),
                "STAGE3_RUN": args.output.name + "_full", "STAGE3_STEPS": "10000",
                "STAGE3_EVAL_EVERY": "100", "STAGE3_SAVE_EVERY": "1000", "STAGE3_SEED": str(args.seed),
                "STAGE3_T_LR": str(selected_c["temperature_lr"]),
                "STAGE3_T_LR_FINAL": str(selected_c["temperature_lr"] * .01),
                "STAGE3_E_LR": str(selected_c["environment_lr"]),
                "STAGE3_E_LR_FINAL": str(selected_c["environment_lr"] * .01),
                "STAGE3_MATERIAL_LR": str(args.material_lr), "STAGE3_MATERIAL_LR_FINAL": str(args.material_lr_final),
                "STAGE3_HOLD": str(args.lr_hold_fraction),
                "STAGE3_LAMBDA_K": str(selected["K"]["regularization"]),
                "STAGE3_LAMBDA_DELTA": str(selected["R"]["regularization"])}
    content = "# Selected on validation only; full matched training has not run.\n"
    content += "".join("export " + name + "=" + shlex.quote(value) + "\n" for name, value in settings.items())
    (args.output / "best_settings.env").write_text(content)
    print("[tune complete] " + json.dumps(selection), flush=True)
    print("[settings] " + str(args.output / "best_settings.env"), flush=True)


if __name__ == "__main__":
    main()
