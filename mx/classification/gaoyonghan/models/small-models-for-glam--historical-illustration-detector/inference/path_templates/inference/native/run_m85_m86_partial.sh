#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python
ROOT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m85_m86_imagenet21k_partial_v1
[[ $# -le 1 && ( $# == 0 || "$1" == '--preflight' ) ]] || { echo '只支持 --preflight 或无参数'; exit 1; }
[[ -x "$PY" ]] || { echo "缺少现有分类环境：$PY"; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo '85/86号部分评测入口已在运行'; exit 1; }
exec > >(tee "$ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
exec "$PY" "$SCRIPT_DIR/run_m85_m86_partial.py" --output-root "$ROOT" \
  --model "${MODEL_NUMBER:-both}" --max-images "${MAX_IMAGES:-0}" "$@"
