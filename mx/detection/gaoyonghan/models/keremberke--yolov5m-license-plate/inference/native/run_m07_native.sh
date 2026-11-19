#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-yolo-v2/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 YOLO_AUTOINSTALL=false
"$PY" -B -c 'import ultralytics; assert ultralytics.__version__ == "8.4.0", "Expected existing Ultralytics 8.4.0 environment"'
# The native COCO validator uses this package; do not upgrade shared dependencies.
if ! "$PY" -B -c 'from importlib.metadata import version; from packaging.version import Version; import faster_coco_eval; assert Version(version("faster-coco-eval")) >= Version("1.6.7")'; then
    "$PY" -m pip install --no-deps --retries 3 --timeout 60 \
        --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" faster-coco-eval==1.6.7
fi
"$PY" -B -u "$SCRIPT_DIR/run_m07_native.py" "$@"
