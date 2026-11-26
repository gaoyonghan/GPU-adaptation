#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv-m83-optimum/bin/python
OUTPUT_ROOT="${OUTPUT_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/results/m83_imagein_source_v1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false USE_TF=0 USE_FLAX=0
[[ $# -le 1 && ( $# == 0 || "$1" == '--preflight' || "$1" == '--describe' ) ]] || {
  echo '支持无参数运行、--preflight或--describe' >&2; exit 1;
}
[[ "$MAX_IMAGES" =~ ^[0-9]+$ ]] || { echo 'MAX_IMAGES必须为非负整数' >&2; exit 1; }
if [[ "${1:-}" == '--describe' ]]; then
  exec /home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python "$SCRIPT_DIR/run_m83_imagein.py" --describe
fi
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/.run.lock"
flock -n 9 || { echo '83号已有任务占用该结果目录' >&2; exit 1; }
exec > >(tee "$OUTPUT_ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
bash "$SCRIPT_DIR/setup_m83_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PY" "$SCRIPT_DIR/run_m83_imagein.py" --output-root "$OUTPUT_ROOT" --max-images "$MAX_IMAGES" "$@"
