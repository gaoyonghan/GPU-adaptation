#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-eval/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4
export YOLO_AUTOINSTALL=false
export MPLCONFIGDIR="$SCRIPT_DIR/../results/m41_omni_opendatalab_four/matplotlib"
mkdir -p "$MPLCONFIGDIR"
"$PY" -B -u "$SCRIPT_DIR/run_m41_omni_four.py" --device "${DEVICE:-0}" --smoke
"$PY" -B -u "$SCRIPT_DIR/run_m41_omni_four.py" --device "${DEVICE:-0}"
