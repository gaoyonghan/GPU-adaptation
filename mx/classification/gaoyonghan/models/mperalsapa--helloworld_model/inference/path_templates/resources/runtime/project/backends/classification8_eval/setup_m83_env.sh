#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
ENV_DIR=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv-m83-optimum
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
[[ -x "$BOOTSTRAP_PYTHON" ]] || { echo "缺少基础环境: $BOOTSTRAP_PYTHON" >&2; exit 1; }
exec 8>"${ENV_DIR}.setup.lock"
flock 8
[[ -x "$ENV_DIR/bin/python" ]] || "$BOOTSTRAP_PYTHON" -m venv --system-site-packages "$ENV_DIR"
PY="$ENV_DIR/bin/python"
if ! "$PY" "$SCRIPT_DIR/run_m83_imagein.py" --check-environment >/dev/null 2>&1; then
  "$PY" -m pip install --index-url https://pypi.org/simple --retries 3 --timeout 60 \
    'optimum[onnxruntime]==1.27.0' 'transformers==4.50.0' \
    'onnx==1.17.0' 'onnxruntime==1.20.1' 'numpy==1.26.4' 'Pillow==11.1.0' \
    'pyarrow==19.0.1' 'scikit-learn==1.5.2' 'datasets==3.4.1' 'protobuf==4.25.6' \
    'huggingface-hub==0.29.3' 'tokenizers==0.21.4' 'torch==2.5.1+cu121'
fi
"$PY" "$SCRIPT_DIR/run_m83_imagein.py" --check-environment
