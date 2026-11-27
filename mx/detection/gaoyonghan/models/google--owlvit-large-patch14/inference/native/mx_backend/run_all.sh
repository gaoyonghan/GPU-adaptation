#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ZERO_SHOT_PYTHON:-/home/ubuntu/gaoyonghan/miniconda3/envs/zero-shot-eval/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/data2/gaoyonghan/gpu/model}"
DATASET_ROOT="${DATASET_ROOT:-/data2/gaoyonghan/gpu/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ubuntu/gaoyonghan/gpu/results/zero_shot_official}"
GPU_IDS="${GPU_IDS:-0,1}"
DTYPE="${DTYPE:-float32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_IMAGES="${MAX_IMAGES:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "找不到环境: $PYTHON_BIN；请先运行 $SCRIPT_DIR/setup_env.sh" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
exec 9<"$OUTPUT_ROOT"
if ! flock -n 9; then
  echo "输出目录正被另一个任务使用: $OUTPUT_ROOT" >&2
  exit 1
fi

COMMON_ARGS=(
  --model-root "$MODEL_ROOT"
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
)
if [[ "$MAX_IMAGES" != "0" ]]; then
  COMMON_ARGS+=(--max-images "$MAX_IMAGES")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/run_model.py" --preflight "${COMMON_ARGS[@]}"

# 两个 DINO 已与官方值对齐，保留在 models.json 和汇总中，默认不重复运行。
GROUP_0=(owlvit-base-patch32 owlv2-large-patch14-ensemble)
GROUP_1=(owlvit-large-patch14 owlv2-base-patch16-ensemble owlv2-base-patch16 omdet-turbo-swin-tiny)

run_group() {
  local gpu_id="$1"
  shift
  local failed=0
  local model_key
  for model_key in "$@"; do
    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$SCRIPT_DIR/run_model.py" \
      --models "$model_key" --device cuda:0 "${COMMON_ARGS[@]}"; then
      failed=1
    fi
  done
  return "$failed"
}

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -eq 1 ]]; then
  set +e
  run_group "${GPUS[0]}" "${GROUP_0[@]}" "${GROUP_1[@]}"
  STATUS=$?
  set -e
  "$PYTHON_BIN" "$SCRIPT_DIR/run_model.py" --summarize --output-root "$OUTPUT_ROOT"
  exit "$STATUS"
fi

set +e
run_group "${GPUS[0]}" "${GROUP_0[@]}" &
PID_0=$!
run_group "${GPUS[1]}" "${GROUP_1[@]}" &
PID_1=$!
wait "$PID_0"
STATUS_0=$?
wait "$PID_1"
STATUS_1=$?
set -e

"$PYTHON_BIN" "$SCRIPT_DIR/run_model.py" --summarize --output-root "$OUTPUT_ROOT"
if [[ $STATUS_0 -ne 0 || $STATUS_1 -ne 0 ]]; then
  echo "至少一个 GPU 队列失败: GPU0=$STATUS_0 GPU1=$STATUS_1" >&2
  exit 1
fi
echo "非 DINO 的 6 个模型已处理完成。"
