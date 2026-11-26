#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
export PYTHONPATH=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nickmuchi--yolos-small-finetuned-license-plate-detection/resources/runtime/assets/workspace/official_object_detection_eval/.runtime/table
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
"$PY" -B -c 'import fitz; assert fitz.VersionBind == "1.21.1", "Requires the existing table PyMuPDF 1.21.1 overlay"'
"$PY" -B -u "$SCRIPT_DIR/run_m15_native_tight.py" --max-images "${MAX_IMAGES:-10}" "$@"
