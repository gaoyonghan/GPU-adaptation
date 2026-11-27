#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
NATIVE_ENV=__MODEL_ROOT__/resources/runtime/assets/workspace/.venv-m94-timm0916
ROOT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m94_native_v1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" OMP_NUM_THREADS=2
[[ $# -le 1 ]] || { echo '仅支持 --describe、--preflight 或无参数'; exit 1; }
if [[ $# == 1 ]]; then
  [[ "$1" == --describe || "$1" == --preflight ]] || { echo '参数不支持'; exit 1; }
  [[ "$1" != --describe ]] || exec "$BASE" -B "$SCRIPT_DIR/run_m94.py" --describe
fi
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo '94号已在运行'; exit 1; }
exec > >(tee "$ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
[[ -x "$NATIVE_ENV/bin/python" ]] || "$BASE" -m venv --system-site-packages "$NATIVE_ENV"
if ! "$NATIVE_ENV/bin/python" - <<'PY'
import importlib.metadata as m
import sys
try:
    ready = m.version('timm') == '0.9.16'
except m.PackageNotFoundError:
    ready = False
sys.exit(0 if ready else 1)
PY
then
  "$NATIVE_ENV/bin/python" -m pip install --no-deps --retries 3 --timeout 60 --index-url https://pypi.org/simple timm==0.9.16 || \
  "$NATIVE_ENV/bin/python" -m pip install --no-deps --retries 3 --timeout 60 --index-url https://pypi.tuna.tsinghua.edu.cn/simple timm==0.9.16
fi
exec "$NATIVE_ENV/bin/python" "$SCRIPT_DIR/run_m94.py" --output-root "$ROOT" --max-images "${MAX_IMAGES:-0}" "$@"
