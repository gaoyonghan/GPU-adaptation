#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python}"
ENV_DIR="$SCRIPT_DIR/.venv"
export PYTHONNOUSERSITE=1
mkdir -p "$SCRIPT_DIR/.runtime"
exec 8>"$SCRIPT_DIR/.runtime/setup.lock"
flock 8
[[ -x "$ENV_DIR/bin/python" ]] || "$BOOTSTRAP_PYTHON" -m venv --system-site-packages "$ENV_DIR"
# Reuse matching base wheels read-only. Missing/different packages install ONLY in this venv.
if ! "$ENV_DIR/bin/python" "$SCRIPT_DIR/run_model.py" --check-environment >/dev/null 2>&1; then
"$ENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 --retries 3 --timeout 30 \
  'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'transformers==4.50.0' \
  'timm==1.0.15' 'numpy==1.26.4' 'Pillow==11.1.0' 'scipy==1.12.0' \
  'scikit-learn==1.5.2' 'onnxruntime==1.20.1' 'safetensors==0.5.3' 'huggingface-hub==0.29.3'
fi
"$ENV_DIR/bin/python" "$SCRIPT_DIR/run_model.py" --setup-sources
