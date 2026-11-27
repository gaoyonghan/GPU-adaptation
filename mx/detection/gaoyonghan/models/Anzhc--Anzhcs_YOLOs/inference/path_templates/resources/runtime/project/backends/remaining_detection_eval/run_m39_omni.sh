#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFER_PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
SCORE_PY=/home/ubuntu/gaoyonghan/miniconda3/envs/omnidocbench/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
unset PYTHONPATH
if [[ "${1:-}" == --preflight ]]; then
    CUDA_VISIBLE_DEVICES="" PYTHONPATH="$SCRIPT_DIR/_ppdeps" "$INFER_PY" -B "$SCRIPT_DIR/run_m39_omni.py" --preflight
    exit
fi
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES仅支持0（10张验证后全量）或10（仅10张）。' >&2; exit 2
fi
OUT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m39_omni_official_v1
mkdir -p "$OUT/logs"
exec 9>"$OUT/.launch.lock"
flock -n 9 || { echo '39号已有任务在运行。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
CUDA_VISIBLE_DEVICES="${DEVICE:-0}" PYTHONPATH="$SCRIPT_DIR/_ppdeps" "$INFER_PY" -B -u "$SCRIPT_DIR/run_m39_omni.py" --max-images 10
CUDA_VISIBLE_DEVICES="" "$SCORE_PY" -B -u "$SCRIPT_DIR/score_m39_omni_official.py" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    CUDA_VISIBLE_DEVICES="${DEVICE:-0}" PYTHONPATH="$SCRIPT_DIR/_ppdeps" "$INFER_PY" -B -u "$SCRIPT_DIR/run_m39_omni.py" --max-images 0
    CUDA_VISIBLE_DEVICES="" "$SCORE_PY" -B -u "$SCRIPT_DIR/score_m39_omni_official.py" --max-images 0
fi
