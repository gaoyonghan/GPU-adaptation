#!/usr/bin/env bash
set -euo pipefail
# Reuse CUDA/PyTorch; isolate the model-card Ultralytics versions.
BASE=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-eval/bin/python
ENV=__MODEL_ROOT__/resources/runtime/assets/workspace/.venv-m12-ultralytics8043
unset PYTHONPATH
export PYTHONNOUSERSITE=1
if [[ ! -x "$ENV/bin/python" ]]; then
    "$BASE" -m venv --system-site-packages "$ENV"
fi
"$ENV/bin/python" -m pip install --retries 3 --timeout 60 \
    --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" --no-deps \
    ultralytics==8.0.43 ultralyticsplus==0.0.28 sentry-sdk==1.45.0 setuptools==79.0.1
"$ENV/bin/python" -B - <<'PY'
import ultralytics
from importlib.metadata import version
from ultralytics.yolo.v8.detect.val import DetectionValidator
assert ultralytics.__version__ == '8.0.43'
assert version('ultralyticsplus') == '0.0.28'
for name in ['ultralytics', 'ultralyticsplus', 'torch', 'torchvision', 'numpy']:
    print(name, version(name))
print('Environment ready; no model loaded or inference started.')
PY
