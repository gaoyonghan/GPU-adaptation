#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ZERO_SHOT_ENV_DIR:-/home/ubuntu/gaoyonghan/miniconda3/envs/zero-shot-eval}"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/gaoyonghan/miniconda3/bin/conda}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "找不到 conda: $CONDA_BIN" >&2
  exit 1
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_DIR" python=3.11 pip
fi

if "$ENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import importlib.metadata as m
required = {
    "torch": "2.5.1", "torchvision": "0.20.1", "transformers": "4.50.0",
    "timm": "1.0.15", "numpy": "1.26.4", "scipy": "1.12.0",
    "Pillow": "11.1.0", "pycocotools": "2.0.8", "lvis": "0.5.3",
    "safetensors": "0.5.3",
}
for package, version in required.items():
    installed = m.version(package)
    if package == "torch":
        installed = installed.split("+")[0]
    assert installed == version, (package, installed, version)
PY
then
  echo "环境依赖已满足，无需重复安装: $ENV_DIR"
else
  "$ENV_DIR/bin/python" -m pip install --index-url "$PYPI_INDEX" --upgrade pip
  "$ENV_DIR/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1
  "$ENV_DIR/bin/python" -m pip install --index-url "$PYPI_INDEX" \
    transformers==4.50.0 timm==1.0.15 numpy==1.26.4 scipy==1.12.0 \
    Pillow==11.1.0 pycocotools==2.0.8 lvis==0.5.3 safetensors==0.5.3
fi

"$ENV_DIR/bin/python" - <<'PY'
import torch, transformers
from transformers import OmDetTurboForObjectDetection, Owlv2ForObjectDetection
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("required model classes: OK")
PY
