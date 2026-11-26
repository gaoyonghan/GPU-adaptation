#!/usr/bin/env bash
set -euo pipefail

# 与安装阶段保持相同的用户级包隔离，子进程也继承此设置。
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${OBJECT_DETECTION_ENV_ROOT:-/home/ubuntu/gaoyonghan/miniconda3/envs}"
MODEL_ROOT="${MODEL_ROOT:-/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-page-elements-v3/resources/model_index}"
DATASET_ROOT="${DATASET_ROOT:-/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets}"
SOURCE_ROOT="${OFFICIAL_SOURCE_ROOT:-/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-page-elements-v3/resources/runtime/assets/workspace/official_sources/object_detection}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-page-elements-v3/resources/runtime/assets/workspace/results/object_detection_official_v2}"
GPU_IDS="${GPU_IDS:-0,1}"
DTYPE="${DTYPE:-float32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_IMAGES="${MAX_IMAGES:-10}"
AUTO_SETUP="${AUTO_SETUP:-1}"

if [[ "$AUTO_SETUP" == "1" ]]; then
  bash "$SCRIPT_DIR/setup_env.sh"
fi

CONTROL_PY="$ENV_ROOT/object-detection-hf-v2/bin/python"
if [[ ! -x "$CONTROL_PY" ]]; then
  echo "缺少控制环境，请运行: bash $SCRIPT_DIR/setup_env.sh" >&2
  exit 1
fi

python_for_profile() {
  case "$1" in
    hf) echo "$ENV_ROOT/object-detection-hf-v2/bin/python" ;;
    yolo) echo "$ENV_ROOT/object-detection-yolo-v2/bin/python" ;;
    yolov5) echo "$ENV_ROOT/object-detection-yolov5-v2/bin/python" ;;
    yolov10) echo "$ENV_ROOT/object-detection-yolov10-v2/bin/python" ;;
    yoloe) echo "$ENV_ROOT/object-detection-yoloe-v2/bin/python" ;;
    yolov13) echo "$ENV_ROOT/object-detection-yolov13-v2/bin/python" ;;
    rfdetr) echo "$ENV_ROOT/object-detection-rfdetr-v2/bin/python" ;;
    tensorflow) echo "$ENV_ROOT/object-detection-tensorflow-v2/bin/python" ;;
    onnx) echo "$ENV_ROOT/object-detection-onnx-v2/bin/python" ;;
    wilddet3d) echo "$ENV_ROOT/object-detection-wilddet3d-v2/bin/python" ;;
    *) echo "未知环境配置: $1" >&2; return 1 ;;
  esac
}

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.run_all.lock"
if ! flock -n 9; then
  echo "输出目录正被另一个批量任务使用: $OUTPUT_ROOT" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_AUTOINSTALL=false
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

COMMON_ARGS=(
  --model-root "$MODEL_ROOT"
  --dataset-root "$DATASET_ROOT"
  --source-root "$SOURCE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
  --max-images "$MAX_IMAGES"
)

"$CONTROL_PY" "$SCRIPT_DIR/run_model.py" --preflight --paths-only "${COMMON_ARGS[@]}"
"$CONTROL_PY" "$SCRIPT_DIR/run_model.py" --materialize-blocked --output-root "$OUTPUT_ROOT"

run_shard() {
  local gpu_id="$1"
  local shard="$2"
  local failed=0
  local model_key profile python_bin library_path library_dir
  while IFS=$'\t' read -r model_key profile; do
    python_bin="$(python_for_profile "$profile")"
    if [[ ! -x "$python_bin" ]]; then
      echo "[$model_key] 缺少环境 $profile: $python_bin" >&2
      failed=1
      continue
    fi
    library_path="${LD_LIBRARY_PATH:-}"
    for library_dir in "${python_bin%/bin/python}"/lib/python*/site-packages/nvidia/*/lib; do
      [[ -d "$library_dir" ]] && library_path="$library_dir:$library_path"
    done
    if ! LD_LIBRARY_PATH="$library_path" CUDA_VISIBLE_DEVICES="$gpu_id" OBJECT_DETECTION_ENV_PROFILE="$profile" \
      "$python_bin" "$SCRIPT_DIR/run_model.py" --models "$model_key" --device cuda:0 "${COMMON_ARGS[@]}"; then
      failed=1
    fi
  done < <("$CONTROL_PY" "$SCRIPT_DIR/run_model.py" --list-runnable-with-env --shard "$shard")
  return "$failed"
}

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_IDS 不能为空" >&2
  exit 1
fi

declare -a PIDS=()
for index in "${!GPUS[@]}"; do
  run_shard "${GPUS[$index]}" "$index/${#GPUS[@]}" &
  PIDS+=("$!")
done

failed=0
set +e
for pid in "${PIDS[@]}"; do
  wait "$pid"
  [[ $? -ne 0 ]] && failed=1
done
set -e

"$CONTROL_PY" "$SCRIPT_DIR/run_model.py" --summarize --output-root "$OUTPUT_ROOT"
if [[ $failed -ne 0 ]]; then
  echo "至少一个 GPU 队列存在实际运行失败；详情见各模型 status.json。" >&2
  exit 1
fi
echo "37 个绿色检测模型的本轮任务已全部处理。"
