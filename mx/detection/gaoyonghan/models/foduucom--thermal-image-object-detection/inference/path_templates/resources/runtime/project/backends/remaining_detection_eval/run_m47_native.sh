#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=__MODEL_ROOT__/resources/runtime/assets/workspace/.venv-m12-ultralytics8043/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 YOLO_AUTOINSTALL=false OMP_NUM_THREADS=4
if [[ "${1:-}" == --preflight ]]; then
    CUDA_VISIBLE_DEVICES="" "$PY" -B "$SCRIPT_DIR/run_m47_native.py" --preflight
    exit
fi
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES仅支持0（先10张后全量）或10（仅10张）。' >&2; exit 2
fi
OUT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m47_native8043_flir8_v1
mkdir -p "$OUT/logs"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '47号已有任务运行。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
"$PY" -B -u "$SCRIPT_DIR/run_m47_native.py" --device "${DEVICE:-0}" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$PY" -B -u "$SCRIPT_DIR/run_m47_native.py" --device "${DEVICE:-0}" --max-images 0
fi
