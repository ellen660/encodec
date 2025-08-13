#!/bin/bash

set -e  # Exit on error

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
  --nproc_per_node=8 \
  encodec/trainer/init_config.py \
  --exp_name baseline_eeg \
  --log_dir "$ROOT_DIR/encodec/ablations/baseline/eeg"

#   Here are solid STFT settings:

# One-pass “good compromise”
# nperseg = 512 (2.0 s window) → Δf = fs/nperseg = 0.5 Hz (resolves delta)

# noverlap = 384 (75% overlap) → hop = 0.5 s

# nfft = 1024 (zero-pad for smoother spectrum)

# window = 'hann'

# Two-pass (often better)
# Low bands (delta–theta):

# nperseg = 1024 (4.0 s) → Δf = 0.25 Hz

# noverlap = 768 (75%) → hop = 1.0 s

# nfft = 2048, window='hann'

# High bands (beta–gamma):

# nperseg = 256 (1.0 s) → Δf = 1 Hz

# noverlap = 192 (75%) → hop = 0.25 s

# nfft = 512, window='hann'