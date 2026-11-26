#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-eval/bin/python
OUT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m37_native_entry_v1
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES仅支持0（先冒烟后全量）或10（仅冒烟）。' >&2
    exit 2
fi
mkdir -p "$OUT/logs" "$OUT/matplotlib"
export MPLCONFIGDIR="$OUT/matplotlib"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '37号已有任务运行，请勿重复启动。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
if [[ "${1:-}" == --preflight ]]; then
    "$PY" -B "$SCRIPT_DIR/run_m37_native.py" --preflight
    exit
fi
"$PY" -B -u "$SCRIPT_DIR/run_m37_native.py" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$PY" -B -u "$SCRIPT_DIR/run_m37_native.py" --max-images 0
fi
