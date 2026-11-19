#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT=/home/ubuntu/gaoyonghan/miniconda3/envs
MODERN_PY="$ENV_ROOT/object-detection-yolo-v2/bin/python"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 YOLO_AUTOINSTALL=false
export YOLOv5_AUTOINSTALL=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OUTPUT_ROOT=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/iitolstykh--YOLO-Face-Person-Detector/resources/runtime/assets/workspace/results/native_18_20_21_v1
mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee "$OUTPUT_ROOT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
# Sequential execution avoids competition among these three jobs. Choose a free GPU via DEVICE.
for number in 18 20 21; do
    case "$number" in
        18) family=yolov5 ;;
        20) family=yolov10 ;;
        21) family=yolo ;;
    esac
    export YOLO_CONFIG_DIR="$OUTPUT_ROOT/.runtime/$family"
    export YOLOV5_CONFIG_DIR="$YOLO_CONFIG_DIR"
    export MPLCONFIGDIR="$OUTPUT_ROOT/.runtime/matplotlib"
    mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
    "$ENV_ROOT/object-detection-$family-v2/bin/python" -B -u \
        "$SCRIPT_DIR/run_native_18_20_21.py" --model-number "$number" --check
done
export YOLO_CONFIG_DIR="$OUTPUT_ROOT/.runtime/yolo"
"$MODERN_PY" -B -u "$SCRIPT_DIR/run_native_18_20_21.py" --prepare
for number in 18 20 21; do
    case "$number" in
        18) family=yolov5 ;;
        20) family=yolov10 ;;
        21) family=yolo ;;
    esac
    export YOLO_CONFIG_DIR="$OUTPUT_ROOT/.runtime/$family"
    export YOLOV5_CONFIG_DIR="$YOLO_CONFIG_DIR"
    if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
        "$ENV_ROOT/object-detection-$family-v2/bin/python" -B -u \
            "$SCRIPT_DIR/run_native_18_20_21.py" --model-number "$number" \
            --device "${DEVICE:-0}" --batch "${BATCH:-16}" --max-images 10
    fi
    "$ENV_ROOT/object-detection-$family-v2/bin/python" -B -u \
        "$SCRIPT_DIR/run_native_18_20_21.py" --model-number "$number" \
        --device "${DEVICE:-0}" --batch "${BATCH:-16}" --max-images "${MAX_IMAGES:-0}"
done
