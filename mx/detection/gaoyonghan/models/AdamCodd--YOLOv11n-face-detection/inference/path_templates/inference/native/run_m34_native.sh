#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=__MODEL_ROOT__/resources/runtime/assets/workspace/.venv-m12-ultralytics8043/bin/python
OUT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m34_native8043_v1
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 YOLO_AUTOINSTALL=false
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES仅支持0（冒烟后全量）或10（仅冒烟）。' >&2
    exit 2
fi
if [[ ! -x "$PY" ]]; then
    echo "缺少原12号环境，请先运行：bash $SCRIPT_DIR/setup_m12_native.sh" >&2
    exit 1
fi
mkdir -p "$OUT/logs" "$OUT/matplotlib"
export MPLCONFIGDIR="$OUT/matplotlib"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '34号已有任务运行，请勿重复启动。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
"$PY" -B "$SCRIPT_DIR/run_m34_native.py" --device "${DEVICE:-0}" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$PY" -B "$SCRIPT_DIR/run_m34_native.py" --device "${DEVICE:-0}" --max-images 0
fi
