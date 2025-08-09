#!/bin/bash

set -e  # Exit on error

# Send email on error
trap 'poetry run python encodec/notify_failure.py' ERR

# Define project root
ROOT_DIR=$(pwd)

# Set PYTHONPATH and run
PYTHONPATH=$ROOT_DIR \
poetry run python encodec/train.py \
  --exp_name baseline_ppg \
  --log_dir "$ROOT_DIR/encodec/ablations/baseline/ppg"
