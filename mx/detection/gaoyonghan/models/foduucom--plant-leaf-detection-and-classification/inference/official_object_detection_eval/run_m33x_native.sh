#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-yolo-v2/bin/python
ENV=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/.venv-m33x-ultralytics83123
OUT=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/results/m33x_native83123_v1
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 YOLO_AUTOINSTALL=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "${1:-}" == --describe ]]; then
    exec "$BASE" -B "$SCRIPT_DIR/run_m33x_native.py" --describe
fi
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES只支持0（先冒烟再全量）或10（仅冒烟）。' >&2; exit 2
fi
export YOLO_CONFIG_DIR="$OUT/.runtime/ultralytics" MPLCONFIGDIR="$OUT/.runtime/matplotlib"
mkdir -p "$OUT/logs" "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '33号x版已有任务运行，请勿重复启动。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
if [[ ! -x "$ENV/bin/python" ]]; then "$BASE" -m venv --system-site-packages "$ENV"; fi
if ! "$ENV/bin/python" -B -c 'from importlib.metadata import version; expected={"ultralytics":"8.3.123","tqdm":"4.67.1","pandas":"2.2.3","seaborn":"0.13.2","py-cpuinfo":"9.0.0","pytz":"2025.2","tzdata":"2025.2"}; assert all(version(k)==v for k,v in expected.items())'; then
    "$ENV/bin/python" -m pip install --no-deps --retries 3 --timeout 60 \
        --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" ultralytics==8.3.123 \
        tqdm==4.67.1 pandas==2.2.3 seaborn==0.13.2 py-cpuinfo==9.0.0 pytz==2025.2 tzdata==2025.2
fi
"$ENV/bin/python" -B -u "$SCRIPT_DIR/run_m33x_native.py" --prepare
"$ENV/bin/python" -B -u "$SCRIPT_DIR/run_m33x_native.py" --device "${DEVICE:-0}" --batch "${BATCH:-16}" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$ENV/bin/python" -B -u "$SCRIPT_DIR/run_m33x_native.py" --device "${DEVICE:-0}" --batch "${BATCH:-16}" --max-images 0
fi
