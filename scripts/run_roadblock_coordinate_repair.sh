#!/usr/bin/env bash
# Fresh Stage2 only. Existing raw data, Stage1 and Stage2 checkpoints are untouched.
set -euo pipefail

GS_WORK="${GS_WORK:-/mnt/sdg/liupengyu/graduate}"
THERMAL_PYTHON="${THERMAL_PYTHON:-python}"
REPAIR_REFERENCE="${REPAIR_REFERENCE:-$GS_WORK/output/RoadBlock_stage2_retry_tv0/thermal_stage2_common.pt}"
REPAIR_SCENE="${REPAIR_SCENE:-$GS_WORK/data/RGBT-Scenes/RGBT-Scenes/RoadBlock}"
REPAIR_TAG="${REPAIR_TAG:-coordfix_v1}"
REPAIR_STEPS="${REPAIR_STEPS:-30000}"
REPAIR_SH_START="${REPAIR_SH_START:-3000}"
REPAIR_SIGNAL_DIR="$REPAIR_SCENE/radiometric_$REPAIR_TAG"
REPAIR_ALIGNMENT="$GS_WORK/output/RoadBlock_${REPAIR_TAG}_alignment.json"
REPAIR_DISPLAY="$GS_WORK/output/RoadBlock_${REPAIR_TAG}_display.json"
REPAIR_OUTPUT="$GS_WORK/output/RoadBlock_stage2_SH2_$REPAIR_TAG"

cd "$GS_WORK"
"$THERMAL_PYTHON" tools/test_thermal_coordinates.py --require_torch
"$THERMAL_PYTHON" tools/prepare_aligned_radiometry.py \
  --reference "$REPAIR_REFERENCE" --scene "$REPAIR_SCENE" \
  --output_dir "$REPAIR_SIGNAL_DIR" --alignment_output "$REPAIR_ALIGNMENT"
"$THERMAL_PYTHON" tools/calibrate_thermal_pseudocolor.py \
  --scene "$REPAIR_SCENE" --checkpoint "$REPAIR_REFERENCE" \
  --radiometric_dir "$REPAIR_SIGNAL_DIR" --allow_new_radiometric \
  --output "$REPAIR_DISPLAY"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$THERMAL_PYTHON" tools/run_sh_residual.py \
  --reference "$REPAIR_REFERENCE" --scene "$REPAIR_SCENE" \
  --radiometric_dir "$REPAIR_SIGNAL_DIR" --allow_new_radiometric \
  --display_calibration "$REPAIR_DISPLAY" --output "$REPAIR_OUTPUT" \
  --degree 2 --steps "$REPAIR_STEPS" --start_step "$REPAIR_SH_START" \
  --bound 0.5 --lr 0.01 --lr_final 0.0001 --regularization 0.001
