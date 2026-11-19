#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python
ENV_DIR=__MODEL_ROOT__/resources/runtime/assets/workspace/.venv-m95-autotrain
ROOT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m95_author_test_v1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://huggingface.co HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=120
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
MODE="${1:---all}"
[[ $# -le 1 && "$MODE" =~ ^--(all|download|preflight|eval)$ ]] || { echo '支持 --all / --download / --preflight / --eval'; exit 2; }
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo '95号已有运行进程'; exit 1; }
exec > >(tee "$ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
if [[ "$MODE" == --all || "$MODE" == --download ]]; then
  "$BASE" -B "$SCRIPT_DIR/run_m95_test.py" --download
  [[ "$MODE" != --download ]] || exit 0
fi
[[ -x "$ENV_DIR/bin/python" ]] || "$BASE" -m venv --system-site-packages "$ENV_DIR"
if ! "$ENV_DIR/bin/python" "$SCRIPT_DIR/run_m95_test.py" --check-environment; then
  "$ENV_DIR/bin/python" -m pip install --retries 3 --timeout 60 --index-url https://pypi.org/simple \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'numpy==1.26.4' \
    'transformers==4.48.0' 'accelerate==1.2.1' 'datasets==3.2.0' \
    'albumentations==1.4.23' 'Pillow==11.0.0' 'scikit-learn==1.6.0' \
    'opencv-python-headless==4.10.0.84' 'huggingface-hub==0.29.3'
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
ARGS=(--output-root "$ROOT" --max-images "${MAX_IMAGES:-0}")
[[ "$MODE" != --preflight ]] || ARGS+=(--preflight)
exec "$ENV_DIR/bin/python" -B "$SCRIPT_DIR/run_m95_test.py" "${ARGS[@]}"
