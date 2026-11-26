#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-wilddet3d-v2/bin/python
OUT=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/runtime/assets/workspace/results/m27_native_text_v1
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --preflight ]]; then
    exec "$PY" -B "$SCRIPT_DIR/run_m27_native.py" --preflight
fi
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES只支持0（先10张冒烟，通过后全量）或10（仅冒烟）。' >&2
    exit 2
fi
mkdir -p "$OUT/logs"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '27号已有任务运行，请勿重复启动。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
"$PY" -B -u "$SCRIPT_DIR/run_m27_native.py" --device "${DEVICE:-1}" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$PY" -B -u "$SCRIPT_DIR/run_m27_native.py" --device "${DEVICE:-1}" --max-images 0
fi
