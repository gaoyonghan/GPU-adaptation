#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv-m83-optimum/bin/python
ROOT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m99_author_source_v1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
[[ $# -le 1 && ( $# == 0 || "$1" == --preflight || "$1" == --describe ) ]] || { echo '支持 --preflight / --describe / 无参数'; exit 2; }
[[ -x "$PY" ]] || { echo "缺少已验证环境: $PY；停止，不自动安装或更换环境。"; exit 1; }
[[ "${1:-}" != --describe ]] || exec "$PY" -B "$SCRIPT_DIR/run_m99.py" --describe
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo '99号已有运行进程'; exit 1; }
exec > >(tee "$ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
exec "$PY" -B "$SCRIPT_DIR/run_m99.py" --output-root "$ROOT" --max-images "${MAX_IMAGES:-0}" "$@"
