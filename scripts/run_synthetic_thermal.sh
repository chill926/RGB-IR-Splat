#!/usr/bin/env bash
set -euo pipefail
GS_WORK="${GS_WORK:-/home/liupengyu/graduate}"
PYTHON="${THERMAL_PYTHON:-/home/liupengyu/miniconda3/envs/3dgs/bin/python}"
SYNTH_DATA="${SYNTH_DATA:-$GS_WORK/data/SyntheticThermal/physical_scene_v1}"
SYNTH_RUN="${SYNTH_RUN:-$GS_WORK/output/SyntheticThermal_run1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONDONTWRITEBYTECODE=1
cd "$GS_WORK"
if [[ ! -f "$SYNTH_DATA/dataset.json" ]]; then
    "$PYTHON" -B tools/generate_thermal_synthetic.py --output "$SYNTH_DATA" --renderer "${SYNTH_RENDERER:-analytic}"
fi
for branch in C R K; do
    if [[ -e "$SYNTH_RUN/$branch" ]]; then
        echo "Existing output $SYNTH_RUN/$branch; choose a new SYNTH_RUN" >&2;exit 1
    fi
done
for branch in C R K; do
    "$PYTHON" -B -u tools/train_thermal_synthetic.py --dataset "$SYNTH_DATA" --output "$SYNTH_RUN/$branch" \
        --branch "$branch" --steps "${SYNTH_STEPS:-2000}" --seed "${SYNTH_SEED:-0}"
    for endpoint in best final; do
        "$PYTHON" -B tools/evaluate_thermal_synthetic.py --checkpoint "$SYNTH_RUN/$branch/$endpoint.pt" --truth --save_images
    done
done
