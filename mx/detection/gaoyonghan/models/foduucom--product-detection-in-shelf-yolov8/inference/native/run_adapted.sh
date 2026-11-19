#!/usr/bin/env bash
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR=/tmp/remaining25-yolo
failed=()
for model in 17 30 34 42 19 22 55; do
    if [[ "$model" == 34 || "$model" == 55 ]]; then
        py=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-eval/bin/python
        export PYTHONPATH="$HERE/_deps"
    else
        py=/home/ubuntu/gaoyonghan/miniconda3/envs/object-detection-hf-v2/bin/python
        unset PYTHONPATH
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
