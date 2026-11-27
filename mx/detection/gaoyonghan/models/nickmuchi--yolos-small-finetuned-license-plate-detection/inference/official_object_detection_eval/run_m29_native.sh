#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-yolov10-v2/bin/python
OUT=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nickmuchi--yolos-small-finetuned-license-plate-detection/resources/runtime/assets/workspace/results/m29_native_v1
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 YOLO_AUTOINSTALL=false
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export YOLO_CONFIG_DIR="$OUT/.runtime/ultralytics" MPLCONFIGDIR="$OUT/.runtime/matplotlib"
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES只支持0（先冒烟再全量）或10（仅冒烟）。' >&2
    exit 2
fi
mkdir -p "$OUT/logs" "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '29号已有任务运行，请勿重复启动。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
if [[ "${1:-}" == --check || "${1:-}" == --describe ]]; then
    exec "$PY" -B -u "$SCRIPT_DIR/run_m29_native.py" --check --batch "${BATCH:-16}"
fi
"$PY" -B -u "$SCRIPT_DIR/run_m29_native.py" --check
"$PY" -B -u "$SCRIPT_DIR/run_m29_native.py" --device "${DEVICE:-0}" --batch "${BATCH:-16}" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$PY" -B -u "$SCRIPT_DIR/run_m29_native.py" --device "${DEVICE:-0}" --batch "${BATCH:-16}" --max-images 0
fi
