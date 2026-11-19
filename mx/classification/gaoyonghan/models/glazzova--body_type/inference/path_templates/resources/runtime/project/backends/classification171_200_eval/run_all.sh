#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
exec __MODEL_ROOT__/resources/runtime/assets/workspace/classification8_eval/.venv/bin/python "$HERE/dispatch.py" --phase "$PHASE" "$@"
