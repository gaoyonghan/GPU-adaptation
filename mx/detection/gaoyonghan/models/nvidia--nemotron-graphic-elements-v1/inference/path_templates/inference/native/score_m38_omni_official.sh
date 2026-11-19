#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/ubuntu/gaoyonghan/miniconda3/envs/omnidocbench/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "${1:-}" == --preflight ]]; then
    exec "$PY" -B "$SCRIPT_DIR/score_m38_omni_official.py" --preflight
fi
OUT=__MODEL_ROOT__/resources/runtime/assets/workspace/results/m38_omni_official_v1
mkdir -p "$OUT/logs"
exec 9>"$OUT/.lock"
flock -n 9 || { echo '38号Omni官方评分已在运行。'; exit 1; }
exec > >(tee "$OUT/logs/launch_$(date +%Y%m%d_%H%M%S)_$$.log") 2>&1
"$PY" -B -u "$SCRIPT_DIR/score_m38_omni_official.py" --max-images 10
"$PY" -B -u "$SCRIPT_DIR/score_m38_omni_official.py" --max-images 0
