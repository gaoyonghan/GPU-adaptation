#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/paddleocr-vl16/bin/python
export PYTHONPATH="$SCRIPT_DIR/_paddledeps"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
if [[ "${1:-}" == --preflight ]]; then
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" -B "$SCRIPT_DIR/run_m38_omni.py" --preflight
fi
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES仅支持0（先10张后全量）或10（仅10张）。' >&2
    exit 2
fi
OUT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_native_v1
mkdir -p "$OUT/logs"
exec 9>"$OUT/.launch.lock"
flock -n 9 || { echo '38号已有运行任务，请勿重复启动。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
"$PY" -B -u "$SCRIPT_DIR/run_m38_omni.py" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$PY" -B -u "$SCRIPT_DIR/run_m38_omni.py" --max-images 0
fi
