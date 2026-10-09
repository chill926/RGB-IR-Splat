#!/usr/bin/env bash
set -euo pipefail
GS_WORK=${GS_WORK:-/mnt/sdg/liupengyu/graduate}
THERMAL_PYTHON=${THERMAL_PYTHON:-python}
D_REFERENCE=${D_REFERENCE:-$GS_WORK/output/RoadBlock_stage2_SH2_coordfix_v1/thermal_stage2_common.pt}
D_SCENE=${D_SCENE:-$GS_WORK/data/RGBT-Scenes/RGBT-Scenes/RoadBlock}
D_SIGNAL=${D_SIGNAL:-$D_SCENE/radiometric_coordfix_v1}
D_DISPLAY=${D_DISPLAY:-$GS_WORK/output/RoadBlock_coordfix_v1_display.json}
D_OUTPUT=${D_OUTPUT:-$GS_WORK/output/RoadBlock_joint_D_v1}
D_STEPS=${D_STEPS:-30000}
D_JOINT_START=${D_JOINT_START:-3000}
cd "$GS_WORK"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$THERMAL_PYTHON" "$GS_WORK/tools/run_joint_thermal.py" \
  --reference "$D_REFERENCE" --scene "$D_SCENE" --radiometric_dir "$D_SIGNAL" \
  --display_calibration "$D_DISPLAY" --output "$D_OUTPUT" \
  --steps "$D_STEPS" --joint_start_step "$D_JOINT_START" --start_step "$D_JOINT_START" \
  --bound 0.5 --lr 0.01 --lr_final 0.0001 --regularization 0.001 \
  --lambda_rgb 1.0 --lambda_signal 1.0 --lambda_display 1.0 \
  --joint_position_lr 0.000016 --joint_position_lr_final 0.00000016 \
  --joint_scaling_lr 0.0005 --joint_rotation_lr 0.0001 \
  --joint_opacity_lr 0.005 --joint_rgb_lr 0.00025
