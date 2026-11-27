#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-__MODEL_ROOT__/resources/runtime/assets/workspace/results/classification8_v1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
GPU_IDS="${GPU_IDS:-0,1}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.run.lock"
flock -n 9 || { echo "此结果目录已有运行任务: $OUTPUT_ROOT" >&2; exit 1; }
[[ "${AUTO_SETUP:-1}" == 0 ]] || bash "$SCRIPT_DIR/setup_env.sh"
PY="$SCRIPT_DIR/.venv/bin/python"
"$PY" "$SCRIPT_DIR/run_model.py" --check-environment
if [[ ! -d "$OUTPUT_ROOT/_source_snapshot" ]]; then
  mkdir "$OUTPUT_ROOT/_source_snapshot"
  cp "$SCRIPT_DIR"/{run_model.py,models.json,run_all.sh,setup_env.sh,README.md} "$OUTPUT_ROOT/_source_snapshot/"
  ln -s "$SCRIPT_DIR/.runtime" "$OUTPUT_ROOT/_source_snapshot/.runtime"
fi
RUNNER="$OUTPUT_ROOT/_source_snapshot/run_model.py"
IFS=, read -r -a gpus <<< "$GPU_IDS"
[[ ${#gpus[@]} -gt 0 ]] || exit 2
for gpu in "${gpus[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPU_IDS必须为数字列表" >&2; exit 2; }; done
models=(71 74 84 87 88 89 91 95)
worker() {
  local slot="$1" index model failed=0
  for ((index=slot; index<${#models[@]}; index+=${#gpus[@]})); do
    model="${models[index]}"
    # Native evaluators have no per-image resume; the small check is kept in its own directory.
    if ! CUDA_VISIBLE_DEVICES="${gpus[slot]}" "$PY" "$RUNNER" --model "$model" \
      --device cuda:0 --max-images 10 --output-root "$OUTPUT_ROOT/smoke" >> "$OUTPUT_ROOT/$model.log" 2>&1; then
      echo "[$model] 冒烟失败，未启动全量；见 $OUTPUT_ROOT/$model.log"; failed=1; continue
    fi
    if [[ "$MAX_IMAGES" != 10 ]]; then
      if ! CUDA_VISIBLE_DEVICES="${gpus[slot]}" "$PY" "$RUNNER" --model "$model" \
        --device cuda:0 --max-images "$MAX_IMAGES" --output-root "$OUTPUT_ROOT" >> "$OUTPUT_ROOT/$model.log" 2>&1; then
        echo "[$model] 失败；见 $OUTPUT_ROOT/$model.log"; failed=1; continue
      fi
    fi
    echo "[$model] 完成当前运行 (MAX_IMAGES=$MAX_IMAGES)"
  done
  return "$failed"
}
pids=()
for ((slot=0; slot<${#gpus[@]}; slot++)); do worker "$slot" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
"$PY" "$RUNNER" --summarize --output-root "$OUTPUT_ROOT/smoke"
if [[ "$MAX_IMAGES" != 10 ]]; then "$PY" "$RUNNER" --summarize --output-root "$OUTPUT_ROOT"; fi
exit "$failed"
