#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python"
DATA_ROOT="${DATA_ROOT:-__DATASET_ROOT__/m76_animedl_test_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/results/m76_animedl_v1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}" HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
[[ -x "$PY" ]] || { echo "缺少已验证环境: $PY" >&2; exit 1; }
[[ "$MAX_IMAGES" =~ ^[0-9]+$ ]] || { echo 'MAX_IMAGES必须为非负整数' >&2; exit 1; }
case "${1:-}" in ''|--plan|--prepare|--preflight) ;; *) echo '只支持--plan、--prepare、--preflight或无参数运行' >&2; exit 1 ;; esac
[[ $# -le 1 ]] || exit 1
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT/logs"
exec 8>"$DATA_ROOT/.run.lock"
flock -n 8 || { echo '此数据目录已有准备或评测任务' >&2; exit 1; }
exec 9>"$OUTPUT_ROOT/.run.lock"
flock -n 9 || { echo '此结果目录已有任务' >&2; exit 1; }
exec > >(tee "$OUTPUT_ROOT/logs/launch_$(date +%Y%m%d_%H%M%S_%N).log") 2>&1
common=(--data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT")
if [[ $# == 1 ]]; then
  "$PY" "$SCRIPT_DIR/run_m76_animedl.py" "${common[@]}" "$1"
  exit
fi
if [[ ! -f "$DATA_ROOT/prepared.json" ]]; then
  echo '全量测试数据尚未准备：请先执行 bash run_m76_animedl.sh --prepare。'
  echo '准备阶段将下载约259GB压缩包；完成后重新运行本评测入口。'
  exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" "$SCRIPT_DIR/run_m76_animedl.py" "${common[@]}" --max-images 10
if [[ "$MAX_IMAGES" != 10 ]]; then
  "$PY" "$SCRIPT_DIR/run_m76_animedl.py" "${common[@]}" --max-images "$MAX_IMAGES"
fi
