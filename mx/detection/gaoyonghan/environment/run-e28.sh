#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 0 ]; then set -- bash; fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec docker run --rm --pull never --network none --shm-size 16g --group-add video --device /dev/dri:/dev/dri --device /dev/mxcd:/dev/mxcd -v "$project_root:/project" -w /project mx-model-envs:20261118 bash -c 'source /opt/model-envs/e28/bin/activate; exec "$@"' bash "$@"
