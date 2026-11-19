#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
NATIVE_ENV=__MODEL_ROOT__/resources/runtime/assets/workspace/.venv-m92-huggingpics
ROOT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m92_huggingpics_native_v1
[[ $# -le 1 ]] || { echo '只支持 --describe、--preflight 或无参数'; exit 1; }
if [[ $# == 1 ]]; then
  [[ "$1" == --describe || "$1" == --preflight ]] || { echo '参数不支持'; exit 1; }
  if [[ "$1" == --describe ]]; then
    exec "$BASE" "$SCRIPT_DIR/run_m92_native.py" --describe
  fi
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" OMP_NUM_THREADS=2
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo '92号原生流程已在运行'; exit 1; }
exec > >(tee "$ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
[[ -x "$NATIVE_ENV/bin/python" ]] || "$BASE" -m venv --system-site-packages "$NATIVE_ENV"
if ! "$NATIVE_ENV/bin/python" -c 'import importlib.metadata as m; assert all(m.version(k)==v for k,v in {"pytorch-lightning":"2.1.2","torchmetrics":"1.2.0","lightning-utilities":"0.10.0"}.items())'; then
  "$NATIVE_ENV/bin/python" -m pip install --no-deps --retries 3 --timeout 60 \
    --index-url https://pypi.org/simple pytorch-lightning==2.1.2 torchmetrics==1.2.0 lightning-utilities==0.10.0 || \
  "$NATIVE_ENV/bin/python" -m pip install --no-deps --retries 3 --timeout 60 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple pytorch-lightning==2.1.2 torchmetrics==1.2.0 lightning-utilities==0.10.0
fi
exec "$NATIVE_ENV/bin/python" "$SCRIPT_DIR/run_m92_native.py" --output-root "$ROOT" --max-images "${MAX_IMAGES:-0}" "$@"
