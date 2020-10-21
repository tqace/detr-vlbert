#!/usr/bin/env bash
python ./scripts/launch.py \
  --master_port 29502 \
  --nproc_per_node "$1" \
  "$2" --cfg "$3" --model-dir "$4"
