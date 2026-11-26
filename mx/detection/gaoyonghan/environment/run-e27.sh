#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 0 ]; then set -- bash; fi
exec docker run --rm --pull never --network none --shm-size 16g --group-add video --device /dev/dri:/dev/dri --device /dev/mxcd:/dev/mxcd -v /mnt/data/shuaige/model_inference_for_adaptation:/project -w /project mx-model-envs:20261124 bash -c 'source /opt/model-envs/e27/bin/activate; exec "$@"' bash "$@"
