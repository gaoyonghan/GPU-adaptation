#!/usr/bin/env bash
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=/tmp/remaining25-yolo
mkdir -p "$YOLO_CONFIG_DIR"
failed=()
for model in 3 9 12 28 46 47 58; do
    if [[ "$model" == 28 ]]; then
        py=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
    else
        py=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-eval/bin/python
    fi
    if "$py" "$HERE/run_model.py" --model "$model" --device cuda:1 --max-images 10; then
        "$py" "$HERE/run_model.py" --model "$model" --device cuda:1 --max-images 0 || failed+=("$model:full")
    else
        failed+=("$model:smoke")
    fi
done
if (( ${#failed[@]} )); then
    printf 'Failed: %s\n' "${failed[@]}"
    exit 1
fi
