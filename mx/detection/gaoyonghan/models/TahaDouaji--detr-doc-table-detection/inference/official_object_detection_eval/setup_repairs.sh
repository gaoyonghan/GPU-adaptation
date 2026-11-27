#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${OBJECT_DETECTION_ENV_ROOT:-/home/ubuntu/gaoyonghan/miniconda3/envs}"
TABLE_RUNTIME="$SCRIPT_DIR/.runtime/table"
PY="$ENV_ROOT/object-detection-hf-v2/bin/python"
# microsoft/table-transformer environment.yml @16d124f specifies PyMuPDF 1.21.1.
# A scoped overlay preserves the environments used to produce the original 31 results.
if ! PYTHONPATH="$TABLE_RUNTIME" "$PY" -c 'import fitz; assert fitz.VersionBind == "1.21.1"' 2>/dev/null; then
  "$PY" -m pip install --no-deps --only-binary=:all: --target "$TABLE_RUNTIME" \
    --index-url https://pypi.org/simple --retries 3 --timeout 30 PyMuPDF==1.21.1
fi
PYTHONPATH="$TABLE_RUNTIME" "$PY" -c 'import fitz,numpy as np; assert fitz.VersionBind == "1.21.1"; assert list(fitz.Rect(np.array([0.,0.,2.,3.]))) == [0.,0.,2.,3.]; print("Official GriTS Rect compatibility: PASS")'
