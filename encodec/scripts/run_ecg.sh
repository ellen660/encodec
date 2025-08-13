#!/bin/bash

set -e  # Exit on error

# Send email on error
trap 'poetry run python encodec/notify_failure.py' ERR
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Define project root
ROOT_DIR=$(pwd)

# Run distributed training with torch.distributed.run (DDP launcher)
PYTHONPATH=$ROOT_DIR \
poetry run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  encodec/trainer/init_config.py \
  --exp_name baseline_ecg \
  --log_dir "$ROOT_DIR/encodec/ablations/baseline/ecg"