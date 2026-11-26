#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-yolov10-v2/bin/python
PREP=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-yolo-v2/bin/python
ENV=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/PaddlePaddle--PP-DocLayoutV3_safetensors/resources/runtime/assets/workspace/.venv-m24-ultralytics8021
OUT=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/PaddlePaddle--PP-DocLayoutV3_safetensors/resources/runtime/assets/workspace/results/m24_native8021_v1
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 YOLO_AUTOINSTALL=false
if [[ "${1:-}" == --describe ]]; then
    exec "$BASE" -B "$SCRIPT_DIR/run_m24_native.py" --describe
fi
if [[ "${MAX_IMAGES:-0}" != 0 && "${MAX_IMAGES:-0}" != 10 ]]; then
    echo 'MAX_IMAGES只支持0（先冒烟再全量）或10（仅冒烟）。' >&2
    exit 2
fi
mkdir -p "$OUT/logs" "$OUT/.runtime/matplotlib"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '24号已有任务运行，请勿重复启动。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
export MPLCONFIGDIR="$OUT/.runtime/matplotlib"
"$BASE" -B -c 'import sys; from importlib.metadata import version; assert sys.version_info[:2] == (3, 9); assert version("torch") == "2.2.2+cu121"; assert version("torchvision") == "0.17.2+cu121"'
if [[ ! -x "$ENV/bin/python" ]]; then
    "$BASE" -m venv --system-site-packages "$ENV"
fi
# No changes to the shared PyTorch/CUDA environment. Only lightweight packages
# are installed locally; ultralyticsplus is a Hub loader, not the evaluator.
if ! "$ENV/bin/python" -B -c 'from importlib.metadata import version; assert version("ultralytics") == "8.0.21"; assert version("setuptools") == "79.0.1"; assert version("ipython") == "8.18.1"; assert version("tensorboard") == "2.18.0"; assert version("sentry-sdk") == "1.45.0"'; then
    "$ENV/bin/python" -m pip install --retries 3 --timeout 60 \
        --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" --no-deps \
        ultralytics==8.0.21 setuptools==79.0.1
    "$ENV/bin/python" -m pip install --retries 3 --timeout 60 \
        --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
        ipython==8.18.1 tensorboard==2.18.0 sentry-sdk==1.45.0
fi
"$PREP" -B -u "$SCRIPT_DIR/run_m24_native.py" --prepare-only
"$ENV/bin/python" -B -u "$SCRIPT_DIR/run_m24_native.py" --preflight
# Same native configuration for smoke and full validation. Smoke is not a result.
"$ENV/bin/python" -B -u "$SCRIPT_DIR/run_m24_native.py" --device "${DEVICE:-0}" --max-images 10
if [[ "${MAX_IMAGES:-0}" == 0 ]]; then
    "$ENV/bin/python" -B -u "$SCRIPT_DIR/run_m24_native.py" --device "${DEVICE:-0}" --max-images 0
fi
