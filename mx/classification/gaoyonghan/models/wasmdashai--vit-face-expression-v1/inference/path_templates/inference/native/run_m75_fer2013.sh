#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python"
FER_CSV="${FER_CSV:-__DATASET_ROOT__/fer2013/fer2013.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/results/m75_fer2013_v1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
[[ -x "$PY" ]] || { echo "缺少已验证的classification8_eval环境: $PY" >&2; exit 1; }
[[ "$MAX_IMAGES" =~ ^[0-9]+$ ]] || { echo 'MAX_IMAGES必须为非负整数' >&2; exit 1; }
common=(--csv "$FER_CSV" --output-root "$OUTPUT_ROOT" --device cuda:0)
if [[ "${1:-}" == '--preflight' ]]; then
  exec "$PY" "$SCRIPT_DIR/run_m75_fer2013.py" "${common[@]}" --preflight
fi
[[ $# == 0 ]] || { echo '仅支持无参数运行或--preflight' >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/.run.lock"
flock -n 9 || { echo '75号已有任务占用此结果目录' >&2; exit 1; }
exec > >(tee "$OUTPUT_ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
"$PY" "$SCRIPT_DIR/run_m75_fer2013.py" "${common[@]}" --max-images 10
if [[ "$MAX_IMAGES" != 10 ]]; then
  "$PY" "$SCRIPT_DIR/run_m75_fer2013.py" "${common[@]}" --max-images "$MAX_IMAGES"
fi
