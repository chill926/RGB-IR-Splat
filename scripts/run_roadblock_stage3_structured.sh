#!/usr/bin/env bash
# Same C/R/K protocol, shared temperature controls and SH2 environment.
set -euo pipefail
export GS_WORK="${GS_WORK:-/home/liupengyu/graduate}"
export STAGE3_RUN="${STAGE3_RUN:-RoadBlock_stage3_structured_run1}"
export STAGE3_COMMON="${STAGE3_COMMON:-$GS_WORK/output/RoadBlock_stage3_D_pilot_run1_common/thermal_stage3_common.pt}"
export STAGE3_STRUCTURED=1
export STAGE3_STEPS="${STRUCTURED_STEPS:-10000}"
export STAGE3_EVAL_EVERY="${STRUCTURED_EVAL_EVERY:-100}"
export STAGE3_T_LR="${STRUCTURED_T_LR:-3e-4}"
export STAGE3_T_LR_FINAL="${STRUCTURED_T_LR_FINAL:-3e-6}"
export STAGE3_E_LR="${STRUCTURED_E_LR:-1e-4}"
export STAGE3_E_LR_FINAL="${STRUCTURED_E_LR_FINAL:-1e-6}"
export STAGE3_MATERIAL_LR="${STRUCTURED_MATERIAL_LR:-1e-4}"
export STAGE3_MATERIAL_LR_FINAL="${STRUCTURED_MATERIAL_LR_FINAL:-1e-6}"
export STAGE3_LAMBDA_K="${STRUCTURED_LAMBDA_K:-1e-5}"
export STAGE3_LAMBDA_DELTA="${STRUCTURED_LAMBDA_DELTA:-1e-5}"
export STAGE3_HOLD="${STRUCTURED_HOLD:-0.1}"
export STAGE3_SPATIAL="${STRUCTURED_SPATIAL:-0.01}"
exec bash "$GS_WORK/scripts/run_roadblock_stage3.sh"
