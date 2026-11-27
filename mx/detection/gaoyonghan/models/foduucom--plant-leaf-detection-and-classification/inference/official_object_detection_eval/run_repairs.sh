#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${OBJECT_DETECTION_ENV_ROOT:-/home/ubuntu/gaoyonghan/miniconda3/envs}"
SOURCE_ROOT="${OFFICIAL_SOURCE_ROOT:-/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/official_sources/object_detection}"
REPAIR_SCOPE="${REPAIR_SCOPE:-all}"
case "$REPAIR_SCOPE" in
  all) DEFAULT_REPAIR_ROOT=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/results/object_detection_official_v3_repairs ;;
  table) DEFAULT_REPAIR_ROOT=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/results/object_detection_official_v4_tables ;;
  *) echo "REPAIR_SCOPE must be all or table" >&2; exit 2 ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_REPAIR_ROOT}"
MAX_IMAGES="${MAX_IMAGES:-10}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=4
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.run_repairs.lock"
flock -n 9 || { echo "已有补跑进程使用 $OUTPUT_ROOT" >&2; exit 1; }
# Freeze new experiments once; never update an existing experiment's code.
RUNNER_DIR="$OUTPUT_ROOT/_source_snapshot"
if [[ ! -d "$RUNNER_DIR" ]]; then
  mkdir "$RUNNER_DIR"
  cp "$SCRIPT_DIR"/{run_model.py,models.json,download_official_source.py,retinanet_official.py,run_repairs.sh,setup_repairs.sh,prepare_wilddet_weights.py,check_table_repair.py,repair_protocol.json} "$RUNNER_DIR/"
  ln -s "$SCRIPT_DIR/.runtime" "$RUNNER_DIR/.runtime"
fi
[[ -f "$RUNNER_DIR/run_model.py" ]] || { echo "源码快照不完整: $RUNNER_DIR" >&2; exit 1; }
if [[ "$REPAIR_SCOPE" == table ]]; then
  previous_cache=/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/runtime/assets/workspace/results/object_detection_official_v3_repairs/_prepared_datasets
  if [[ ! -e "$OUTPUT_ROOT/_prepared_datasets" && -d "$previous_cache" ]]; then
    ln -s "$previous_cache" "$OUTPUT_ROOT/_prepared_datasets"
  fi
fi

run_queue() {
  local gpu="$1" item key profile py library_dir library_path module_path failed=0
  shift
  for item in "$@"; do
    key="${item%:*}"; profile="${item##*:}"
    py="$ENV_ROOT/object-detection-$profile-v2/bin/python"
    library_path="${LD_LIBRARY_PATH:-}"
    for library_dir in "${py%/bin/python}"/lib/python*/site-packages/nvidia/*/lib; do
      [[ -d "$library_dir" ]] && library_path="$library_dir:$library_path"
    done
    module_path="${PYTHONPATH:-}"
    [[ "$profile" == hf ]] && module_path="$SCRIPT_DIR/.runtime/table:$module_path"
    if ! CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH="$library_path" PYTHONPATH="$module_path" \
      OBJECT_DETECTION_ENV_PROFILE="$profile" "$py" "$RUNNER_DIR/run_model.py" \
      --models "$key" --device cuda:0 --max-images "$MAX_IMAGES" --dtype float32 \
      --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" \
      --model-root "${MODEL_ROOT:-/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/foduucom--plant-leaf-detection-and-classification/resources/model_index}" \
      --dataset-root "${DATASET_ROOT:-/home/ubuntu/文档/ChatGPT/idea 2/outputs/launch_audit_20261005/migration_inputs/datasets}" \
      >> "$OUTPUT_ROOT/$key.log" 2>&1; then
      echo "[$key] 失败；见 $OUTPUT_ROOT/$key.log" >&2; failed=1
    else
      echo "[$key] 本轮完成 (MAX_IMAGES=$MAX_IMAGES)"
    fi
  done
  return "$failed"
}
if [[ "$REPAIR_SCOPE" == table ]]; then
  run_queue 0 m06-table-transformer-structure-recognition:hf &
  first=$!
  run_queue 1 m15-table-transformer-structure-recognition-v1-1-all:hf &
  second=$!
else
  run_queue 0 m53-detr-resnet-50:onnx m23-yoloe:yoloe m06-table-transformer-structure-recognition:hf &
  first=$!
  run_queue 1 m51-object-detection-retinanet:tensorflow m27-wilddet3d:wilddet3d m15-table-transformer-structure-recognition-v1-1-all:hf &
  second=$!
fi
failed=0
wait "$first" || failed=1
wait "$second" || failed=1
exit "$failed"
