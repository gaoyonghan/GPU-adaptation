#!/usr/bin/env bash
set -euo pipefail
cd __MODEL_ROOT__/resources/runtime/assets/workspace
MODEL30_DIR="$PWD/remaining_detection_eval/supplemental_20260905/model30"
MODEL30_PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
"$MODEL30_PY" -m pip install --index-url https://pypi.org/simple --target "$MODEL30_DIR/_deps" --no-deps -r "$MODEL30_DIR/requirements-private.txt"
python -u "$MODEL30_DIR/download.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1
"$MODEL30_PY" -u "$MODEL30_DIR/run.py" --max-images 10
"$MODEL30_PY" -u "$MODEL30_DIR/run.py" --max-images 0
"$MODEL30_PY" -u "$MODEL30_DIR/run_batch8.py" --max-images 16
"$MODEL30_PY" -u "$MODEL30_DIR/run_batch8.py" --max-images 0
python "$MODEL30_DIR/verify.py"
python "$MODEL30_DIR/verify_batch8.py"
