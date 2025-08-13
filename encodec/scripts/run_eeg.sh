#!/bin/bash

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Send email on error
trap 'poetry run python encodec/notify_failure.py' ERR

# Define project root
ROOT_DIR=$(pwd)

# # Set PYTHONPATH and run
# PYTHONPATH=$ROOT_DIR \
# poetry run python encodec/train.py \
#   --exp_name baseline_eeg \
#   --log_dir "$ROOT_DIR/encodec/ablations/baseline/eeg"

# Run distributed training with torch.distributed.run (DDP launcher)
PYTHONPATH=$ROOT_DIR \
poetry run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  encodec/trainer/init_config.py \
  --exp_name baseline_eeg \
  --log_dir "$ROOT_DIR/encodec/ablations/baseline/eeg"