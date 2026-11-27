#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python
ROOT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m92_faceshape_v1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" OMP_NUM_THREADS=2
[[ -x "$PY" ]] || { echo "缺少既有共享环境: $PY"; exit 1; }
[[ $# -le 1 ]] || { echo '只支持 --describe、--preflight 或无参数'; exit 1; }
if [[ $# == 1 ]]; then
  [[ "$1" == --describe || "$1" == --preflight ]] || { echo '参数不支持'; exit 1; }
  exec "$PY" "$SCRIPT_DIR/run_m92.py" "$1"
fi
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo '92号已在运行'; exit 1; }
exec > >(tee "$ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
exec "$PY" "$SCRIPT_DIR/run_m92.py" --output-root "$ROOT" --max-images "${MAX_IMAGES:-0}"
