#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Read-only reuse of the already validated classification environment; no package upgrades.
PY=__MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python
export PYTHONNOUSERSITE=1
[[ -x "$PY" ]] || { echo "缺少已配置的 classification8_eval 环境，请先恢复该环境。" >&2; exit 1; }
"$PY" "$SCRIPT_DIR/run_model.py" --check-environment
"$PY" "$SCRIPT_DIR/run_model.py" --self-test
