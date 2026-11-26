#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_ROOT=/home/ubuntu/gaoyonghan/miniconda3/envs
PY311="$ENV_ROOT/object-detection-eval/bin/python"
PY310="$ENV_ROOT/paddleocr-vl16/bin/python"
"$PY311" -m pip install --index-url https://pypi.org/simple --target "$HERE/_deps" --no-deps dill==0.3.8 pyarrow==19.0.1 onnxruntime==1.23.2 Cython==3.0.12 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3
"$PY311" -m pip install --index-url https://pypi.org/simple --target "$HERE/_ppdeps" --no-deps transformers==5.3.0 tokenizers==0.22.2 huggingface-hub==1.5.0 safetensors==0.7.0 httpx==0.28.1 httpcore==1.0.9 anyio==4.15.0 idna==3.19 h11==0.16.0 certifi==2026.7.22 typing_extensions==4.16.0 opencv-python-headless==4.10.0.84
"$PY310" -m pip install --index-url https://pypi.org/simple --target "$HERE/_paddledeps" --no-deps pycocotools==2.0.11
"$PY311" "$HERE/apply_pp_patch.py"
cd "$HERE/assets/widerface_evaluator"
PYTHONPATH="$HERE/_deps" "$PY311" setup.py build_ext --inplace
