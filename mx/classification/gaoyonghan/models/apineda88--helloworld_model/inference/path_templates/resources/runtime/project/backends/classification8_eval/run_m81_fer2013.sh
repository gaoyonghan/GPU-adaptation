#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python"
FER_CSV="${FER_CSV:-__DATASET_ROOT__/fer2013/deadskull7/fer2013.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/results/m81_fer2013_v1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
[[ -x "$PY" ]] || { echo "缺少共享环境: $PY" >&2; exit 1; }
[[ "$MAX_IMAGES" =~ ^[0-9]+$ ]] || { echo 'MAX_IMAGES必须为非负整数' >&2; exit 1; }
common=(--csv "$FER_CSV" --output-root "$OUTPUT_ROOT" --device cuda:0)
if [[ $# == 1 && ( "$1" == '--preflight' || "$1" == '--describe' ) ]]; then
  exec "$PY" "$SCRIPT_DIR/run_m81_fer2013.py" "${common[@]}" "$1"
fi
[[ $# == 0 ]] || { echo '仅支持无参数运行、--preflight或--describe' >&2; exit 1; }
[[ "${ALLOW_UNVERIFIED_LABELS:-0}" == 1 ]] || {
  echo '类别映射待核实。仅在接受临时结果时设置ALLOW_UNVERIFIED_LABELS=1；详见README_m81.md。' >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/.run.lock"
flock -n 9 || { echo '81号已有任务占用此结果目录' >&2; exit 1; }
exec > >(tee "$OUTPUT_ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
"$PY" "$SCRIPT_DIR/run_m81_fer2013.py" "${common[@]}" --allow-unverified-labels --max-images 10
if [[ "$MAX_IMAGES" != 10 ]]; then
  "$PY" "$SCRIPT_DIR/run_m81_fer2013.py" "${common[@]}" --allow-unverified-labels --max-images "$MAX_IMAGES"
fi
