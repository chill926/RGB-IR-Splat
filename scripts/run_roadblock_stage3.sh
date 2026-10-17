#!/usr/bin/env bash
# Frozen D C/R/K pilot. Does not overwrite existing common/models.
set -euo pipefail
GS_WORK="${GS_WORK:-/home/liupengyu/graduate}"
THERMAL_PYTHON="${THERMAL_PYTHON:-/home/liupengyu/miniconda3/envs/3dgs/bin/python}"
STAGE3_RUN="${STAGE3_RUN:-RoadBlock_stage3_D_lrpilot_run1}"
STAGE3_STEPS="${STAGE3_STEPS:-2000}"
STAGE3_EVAL_EVERY="${STAGE3_EVAL_EVERY:-100}"
STAGE3_SAVE_EVERY="${STAGE3_SAVE_EVERY:-1000}"
STAGE3_SEED="${STAGE3_SEED:-0}"
STAGE3_T_LR="${STAGE3_T_LR:-1e-4}"
STAGE3_T_LR_FINAL="${STAGE3_T_LR_FINAL:-1e-6}"
STAGE3_E_LR="${STAGE3_E_LR:-1e-5}"
STAGE3_E_LR_FINAL="${STAGE3_E_LR_FINAL:-1e-7}"
STAGE3_MATERIAL_LR="${STAGE3_MATERIAL_LR:-1e-4}"
STAGE3_MATERIAL_LR_FINAL="${STAGE3_MATERIAL_LR_FINAL:-1e-6}"
STAGE3_HOLD="${STAGE3_HOLD:-0.1}"
STAGE3_LAMBDA_K="${STAGE3_LAMBDA_K:-1e-4}"
STAGE3_LAMBDA_DELTA="${STAGE3_LAMBDA_DELTA:-1e-4}"
STAGE3_SPATIAL="${STAGE3_SPATIAL:-0}"
STAGE3_STRUCTURED="${STAGE3_STRUCTURED:-0}"
STAGE3_LEARN_MATERIALS="${STAGE3_LEARN_MATERIALS:-matte_plastic}"
STAGE3_D_CHECKPOINT="${STAGE3_D_CHECKPOINT:-$GS_WORK/output/RoadBlock_joint_D_3090_repro_2080_run1/thermal_joint_D_selected.pt}"
STAGE3_COMMON="${STAGE3_COMMON:-$GS_WORK/output/${STAGE3_RUN}_common/thermal_stage3_common.pt}"
STAGE3_LOCK="$GS_WORK/output/${STAGE3_RUN}_comparison.json"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
cd "$GS_WORK"
structured_args=()
if [[ "$STAGE3_STRUCTURED" == 1 ]]; then
    structured_args=(--structured_field
        --temperature_controls "${STAGE3_T_CONTROLS:-4096}"
        --temperature_control_bound_K "${STAGE3_T_CONTROL_BOUND:-5}"
        --temperature_residual_bound_K "${STAGE3_T_RESIDUAL_BOUND:-0.5}"
        --temperature_residual_sigma_K "${STAGE3_T_RESIDUAL_SIGMA:-0.1}"
        --temperature_residual_lr "${STAGE3_T_RESIDUAL_LR:-1e-4}"
        --temperature_residual_lr_final "${STAGE3_T_RESIDUAL_LR_FINAL:-1e-6}"
        --lambda_temperature_residual "${STAGE3_T_RESIDUAL_PRIOR:-1e-3}"
        --lambda_temperature_control "${STAGE3_T_CONTROL_PRIOR:-1e-4}"
        --environment_sh_bound "${STAGE3_E_SH_BOUND:-0.01}"
        --environment_sh_lr "${STAGE3_E_SH_LR:-1e-3}"
        --environment_sh_lr_final "${STAGE3_E_SH_LR_FINAL:-1e-5}"
        --lambda_environment_sh "${STAGE3_E_SH_PRIOR:-1e-4}")
fi
for branch in C R K; do
    if [[ -e "$GS_WORK/output/${STAGE3_RUN}_${branch}" ]]; then
        echo "Existing output: ${STAGE3_RUN}_${branch}; choose a new STAGE3_RUN" >&2
        exit 1
    fi
done
"$THERMAL_PYTHON" -B tools/export_stage3_common.py --checkpoint "$STAGE3_D_CHECKPOINT" \
    --output "$STAGE3_COMMON" --learn_materials "$STAGE3_LEARN_MATERIALS" --reuse_if_matching
for branch in C R K; do
    "$THERMAL_PYTHON" -B -u train_thermal_stage3.py --common "$STAGE3_COMMON" --branch "$branch" \
        --comparison_protocol "$STAGE3_LOCK" -m "$GS_WORK/output/${STAGE3_RUN}_${branch}" \
        --steps "$STAGE3_STEPS" --seed "$STAGE3_SEED" \
        --eval_every "$STAGE3_EVAL_EVERY" --save_every "$STAGE3_SAVE_EVERY" \
        --temperature_lr "$STAGE3_T_LR" --temperature_lr_final "$STAGE3_T_LR_FINAL" \
        --environment_lr "$STAGE3_E_LR" --environment_lr_final "$STAGE3_E_LR_FINAL" \
        --material_lr "$STAGE3_MATERIAL_LR" --material_lr_final "$STAGE3_MATERIAL_LR_FINAL" \
        --lr_hold_fraction "$STAGE3_HOLD" --lambda_k "$STAGE3_LAMBDA_K" \
        --lambda_delta_epsilon "$STAGE3_LAMBDA_DELTA" --lambda_spatial "$STAGE3_SPATIAL" "${structured_args[@]}"
    "$THERMAL_PYTHON" -B tools/evaluate_thermal_physics.py \
        --checkpoint "$GS_WORK/output/${STAGE3_RUN}_${branch}/thermal_${branch}_best.pt" \
        --modality thermal --thermal_output pseudocolor --split all
    "$THERMAL_PYTHON" -B tools/evaluate_thermal_physics.py \
        --checkpoint "$GS_WORK/output/${STAGE3_RUN}_${branch}/thermal_${branch}_best.pt" \
        --modality rgb --split all
    if [[ "$STAGE3_STRUCTURED" == 1 ]]; then
        "$THERMAL_PYTHON" -B tools/evaluate_thermal_physics.py \
            --checkpoint "$GS_WORK/output/${STAGE3_RUN}_${branch}/thermal_${branch}_final.pt" \
            --modality thermal --thermal_output pseudocolor --split all
    fi
done
