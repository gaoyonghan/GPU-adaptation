#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
NATIVE_ENV=__MODEL_ROOT__/resources/runtime/assets/workspace/.venv-m85-m86-swin-native
ROOT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m85_m86_swin_native_v1
[[ $# -le 1 && ( $# == 0 || "$1" == '--preflight' ) ]] || { echo '只支持 --preflight 或无参数'; exit 1; }
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo '原生85/86流程已在运行'; exit 1; }
exec > >(tee "$ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" LOCAL_RANK=0 OMP_NUM_THREADS=2
[[ -x "$NATIVE_ENV/bin/python" ]] || "$BASE" -m venv --system-site-packages "$NATIVE_ENV"
# Reuse existing torch/CUDA, but isolate the two legacy Python dependencies.
if ! "$NATIVE_ENV/bin/python" -c 'import importlib.metadata as m; assert m.version("timm")=="0.4.12" and m.version("yacs")=="0.1.8"'; then
  "$NATIVE_ENV/bin/python" -m pip install --no-deps --retries 3 --timeout 60 \
    --index-url https://pypi.org/simple timm==0.4.12 yacs==0.1.8 || \
  "$NATIVE_ENV/bin/python" -m pip install --no-deps --retries 3 --timeout 60 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple timm==0.4.12 yacs==0.1.8
fi
exec "$NATIVE_ENV/bin/python" "$SCRIPT_DIR/run_m85_m86_native.py" --output-root "$ROOT" \
  --model "${MODEL_NUMBER:-both}" --max-images "${MAX_IMAGES:-0}" "$@"
