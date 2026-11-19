#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SHARED_PY=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python
PY="${EVAL_PYTHON:-$SHARED_PY}"
[[ -x "$PY" ]] || PY="$SCRIPT_DIR/.venv/bin/python"
export PYTHONNOUSERSITE=1
mkdir -p "$SCRIPT_DIR/.runtime"
exec 8>"$SCRIPT_DIR/.runtime/setup.lock"
flock 8
if [[ "$PY" != "$SCRIPT_DIR/.venv/bin/python" ]]; then
  # Existing shared environments are read-only: never pip install into the first batch.
  "$PY" "$SCRIPT_DIR/run_model.py" --check-environment
  exit 0
fi
[[ -x "$PY" ]] || /home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python -m venv --system-site-packages "$SCRIPT_DIR/.venv"
if ! "$PY" "$SCRIPT_DIR/run_model.py" --check-environment >/dev/null 2>&1; then
  "$PY" -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 \
    --retries 3 --timeout 30 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'transformers==4.50.0' \
    'numpy==1.26.4' 'Pillow==11.1.0' 'scikit-learn==1.5.2' 'safetensors==0.5.3' 'huggingface-hub==0.29.3'
fi
"$PY" "$SCRIPT_DIR/run_model.py" --check-environment
