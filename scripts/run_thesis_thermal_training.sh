#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 RGBT_SCENE OUTPUT_ROOT MATERIAL_MASK_DIR MATERIAL_CONFIG [REGULARIZATION_SCAN_JSON]"
  exit 2
fi

SCENE_PATH="$1"
OUTPUT_ROOT="$2"
MATERIAL_MASK_DIR="$3"
MATERIAL_CONFIG="$4"
REGULARIZATION_PROTOCOL="${5:-}"
STAGE1_DIR="${OUTPUT_ROOT}/stage1_rgb"
STAGE2_DIR="${OUTPUT_ROOT}/stage2_constant"
COMMON_ARGS=(-s "$SCENE_PATH" --data_branch rgbt --eval)

python train.py "${COMMON_ARGS[@]}" -m "$STAGE1_DIR" \
  --rgb_geometry_stage --iterations 30000

python train_thermal_image_baseline.py "${COMMON_ARGS[@]}" -m "${OUTPUT_ROOT}/branch_B0" \
  --geometry_model "$STAGE1_DIR" --geometry_iteration -1 --steps 5000 --seed 0

python train_thermal_physics.py "${COMMON_ARGS[@]}" -m "$STAGE2_DIR" \
  --stage stage2 --geometry_model "$STAGE1_DIR" --geometry_iteration -1 \
  --material_mask_dir "$MATERIAL_MASK_DIR" --material_config "$MATERIAL_CONFIG" \
  --steps 10000 --seed 0

STAGE2_CHECKPOINT="${STAGE2_DIR}/thermal_stage2_common.pt"
COMPARISON_PROTOCOL="${OUTPUT_ROOT}/stage3_comparison_protocol.json"

for BRANCH in C K R; do
  BRANCH_ARGS=(--lambda_k 1e-4 --lambda_delta_epsilon 1e-4)
  if [[ -n "$REGULARIZATION_PROTOCOL" ]]; then
    BRANCH_ARGS=(--regularization_protocol "$REGULARIZATION_PROTOCOL")
  fi
  python train_thermal_physics.py "${COMMON_ARGS[@]}" -m "${OUTPUT_ROOT}/branch_${BRANCH}" \
    --stage branch --branch "$BRANCH" --geometry_model "$STAGE1_DIR" --geometry_iteration -1 \
    --stage2_checkpoint "$STAGE2_CHECKPOINT" --steps 5000 --seed 0 \
    --comparison_protocol "$COMPARISON_PROTOCOL" \
    "${BRANCH_ARGS[@]}"
done

python tools/summarize_thermal_branches.py \
  --experiment_root "$OUTPUT_ROOT" --output "${OUTPUT_ROOT}/thermal_branch_comparison.json"
