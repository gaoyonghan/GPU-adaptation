#!/usr/bin/env bash
set -euo pipefail
TASK_DATA=__DATASET_ROOT__/supplemental_20260905/model19
mkdir -p "$TASK_DATA"
aria2c -Z --continue=true --max-connection-per-server=4 --split=4 \
  --min-split-size=8M --connect-timeout=20 --timeout=30 --max-tries=8 \
  --retry-wait=2 --summary-interval=30 --show-console-readout=false \
  --dir="$TASK_DATA" \
  'https://ds4sd-icdar23-doclaynet-competition.s3.eu-de.cloud-object-storage.appdomain.cloud/competition-dataset-public.zip' \
  'https://ds4sd-icdar23-doclaynet-competition.s3.eu-de.cloud-object-storage.appdomain.cloud/competition_dataset_groundtruth.zip'
