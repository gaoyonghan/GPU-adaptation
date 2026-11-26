#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-tensorflow-v2/bin/python
ASSETS=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.runtime/m88_keras
OUTPUT_ROOT="${OUTPUT_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/results/m88_keras_native_v1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
[[ $# -le 1 && ( $# == 0 || "$1" == '--preflight' ) ]] || { echo '只支持 --preflight 或无参数'; exit 1; }
[[ "$MAX_IMAGES" =~ ^[0-9]+$ ]] || { echo 'MAX_IMAGES必须是非负整数'; exit 1; }
[[ -x "$PY" ]] || { echo "缺少现有TensorFlow环境：$PY"; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" KERAS_HOME="$ASSETS"
export TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2
library_path="${LD_LIBRARY_PATH:-}"
for library_dir in "${PY%/bin/python}"/lib/python*/site-packages/nvidia/*/lib; do
    [[ ! -d "$library_dir" ]] || library_path="$library_dir${library_path:+:$library_path}"
done
export LD_LIBRARY_PATH="$library_path"
mkdir -p "$ASSETS/models" "$OUTPUT_ROOT/logs"
exec 9>"$ASSETS/.run.lock"
flock -n 9 || { echo '88号Keras流程已在运行'; exit 1; }
exec > >(tee "$OUTPUT_ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
"$PY" "$SCRIPT_DIR/run_m88_keras.py" --check-environment

download_checked() {
    local url="$1" path="$2" algorithm="$3" expected="$4" actual
    if [[ -f "$path" ]]; then
        actual=$("${algorithm}sum" "$path"); actual="${actual%% *}"
        [[ "$actual" == "$expected" ]] || { echo "缓存校验失败，不覆盖：$path"; exit 1; }
        return
    fi
    curl --fail --location --continue-at - --retry 5 --connect-timeout 30 --max-time 1800 \
        "$url" -o "$path.part"
    actual=$("${algorithm}sum" "$path.part"); actual="${actual%% *}"
    [[ "$actual" == "$expected" ]] || { echo "下载校验失败：$path.part"; exit 1; }
    mv "$path.part" "$path"
}
download_checked 'https://storage.googleapis.com/keras-applications/efficientnetb3.h5' \
    "$ASSETS/models/efficientnetb3.h5" md5 ffd1fdc53d0ce67064dc6a9c7960ede0
download_checked 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json' \
    "$ASSETS/imagenet_class_index.json" sha256 a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc
download_checked 'https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/preprocessing.py' \
    "$ASSETS/preprocessing.py" sha256 8bd9f09079a59af79bb3533892aa14feae9539d7aee121945f5a33dcb3395f9b
exec "$PY" "$SCRIPT_DIR/run_m88_keras.py" --assets "$ASSETS" --output-root "$OUTPUT_ROOT" --max-images "$MAX_IMAGES" "$@"
