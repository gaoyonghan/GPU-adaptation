#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python
SOURCE=__MODEL_ROOT__/resources/runtime/assets/workspace/171-200/inference_code
MODE="${1:-shared_hf}"
mkdir -p "$HERE/.runtime"
exec 9>"$HERE/.runtime/setup-$MODE.lock"
flock 9
case "$MODE" in
  shared_hf)
    "$BASE" -c 'import torch, transformers, pyarrow, sklearn; assert torch.__version__=="2.5.1+cu121"; assert transformers.__version__=="4.50.0"; print("Existing shared environment verified; unchanged.")'
    ;;
  new_hf|yolo)
    ENV_DIR="$HERE/.venv-$MODE"
    [[ -x "$ENV_DIR/bin/python" ]] || "$BASE" -m venv --system-site-packages "$ENV_DIR"
    if [[ -f "$HERE/.runtime/$MODE.freeze.txt" ]]; then
      "$ENV_DIR/bin/python" -m pip freeze | diff - "$HERE/.runtime/$MODE.freeze.txt"
      echo "$MODE environment already frozen; unchanged."
      exit 0
    fi
    if [[ "$MODE" == new_hf ]]; then
      "$ENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple --no-build-isolation "$SOURCE/transformers"
    else
      "$ENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'numpy==1.26.4' 'opencv-python==4.11.0.86' 'ultralytics-thop==2.0.18'
      "$ENV_DIR/bin/python" -m pip install --no-deps --no-build-isolation "$SOURCE/ultralytics"
    fi
    "$ENV_DIR/bin/python" -m pip freeze > "$HERE/.runtime/$MODE.freeze.txt"
    ;;
  mamba)
    ENV_DIR="$HERE/.venv-mamba"
    [[ -x "$ENV_DIR/bin/python" ]] || "$BASE" -m venv --system-site-packages "$ENV_DIR"
    if [[ -f "$HERE/.runtime/mamba.freeze.txt" ]]; then
      "$ENV_DIR/bin/python" -m pip freeze | diff - "$HERE/.runtime/mamba.freeze.txt"
      "$ENV_DIR/bin/python" -c 'import torch, mamba_ssm; assert torch.__version__=="2.6.0+cu124"'
      echo 'Mamba environment already frozen; unchanged.'
      exit 0
    fi
    "$ENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124'
    # CUDA packages shared with the base environment have the required exact
    # versions, but the new nvidia namespace hides their library directories.
    "$ENV_DIR/bin/python" - "$ENV_DIR" <<'PY'
from pathlib import Path
import sys
target=Path(sys.argv[1])/'lib/python3.11/site-packages/nvidia'
source=Path('/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/lib/python3.11/site-packages/nvidia')
for name in ['cudnn','nccl']:
    if not (target/name).exists(): (target/name).symlink_to(source/name,target_is_directory=True)
PY
    "$ENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'transformers==4.50.0' 'timm==1.0.15' 'einops==0.8.1' 'ninja==1.11.1.4'
    "$ENV_DIR/bin/python" -m pip install --no-deps 'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
    "$ENV_DIR/bin/python" -c 'import torch, mamba_ssm; assert torch.__version__=="2.6.0+cu124"; assert not torch._C._GLIBCXX_USE_CXX11_ABI'
    "$ENV_DIR/bin/python" -m pip freeze > "$HERE/.runtime/mamba.freeze.txt"
    ;;
  js)
    cd "$HERE"
    if [[ -f package-lock.json ]]; then ONNXRUNTIME_NODE_INSTALL=skip npm --prefix "$HERE" ci --no-audit --no-fund; else ONNXRUNTIME_NODE_INSTALL=skip npm --prefix "$HERE" install --no-audit --no-fund; fi
    ;;
  all)
    # Release the lock before invoking individual setup modes.
    flock -u 9
    for FAMILY in shared_hf new_hf yolo js mamba; do bash "$HERE/setup_env.sh" "$FAMILY"; done
    ;;
  *) echo "Usage: bash setup_env.sh [shared_hf|new_hf|yolo|js|mamba|all]" >&2; exit 2 ;;
esac
