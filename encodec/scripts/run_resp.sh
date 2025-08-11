#!/bin/bash

set -e  # Exit on error

# Send email on error
trap 'poetry run python encodec/notify_failure.py' ERR

# Define project root
ROOT_DIR=$(pwd)

# # Set PYTHONPATH and run
# PYTHONPATH=$ROOT_DIR \
# poetry run python encodec/trainer/init_config.py \
#   --exp_name baseline_ppg \
#   --log_dir "$ROOT_DIR/encodec/ablations/baseline/ppg"


# Run distributed training with torch.distributed.run (DDP launcher)
PYTHONPATH=$ROOT_DIR \
poetry run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  encodec/trainer/init_config.py \
  --exp_name baseline_resp \
  --log_dir "$ROOT_DIR/encodec/ablations/baseline/resp"