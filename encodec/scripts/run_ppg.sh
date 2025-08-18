#!/bin/bash

set -e  # Exit on error

# Send email on error
trap 'echo "[ERROR] Rank $NODE_RANK failed, killing all processes" && pkill -f torchrun' ERR INT TERM

# ----------------------
# USER CONFIG
# ----------------------
ROOT_DIR=$(pwd)                 # root of your project
SCRIPT=encodec/trainer/init_config.py
NNODES=1                        # total number of nodes
NPROC_PER_NODE=1                # number of GPUs per node
MASTER_ADDR=128.30.202.27           # IP of the master node
# On the master node
# hostname -I | awk '{print $1}'   # prints the primary IP
MASTER_PORT=29501               # TCP port for DDP communication
# check if free: lsof -i :29500
NODE_RANK=$1                    # pass 0 for master, 1..N-1 for workers

# ----------------------
# RUN
# ----------------------
export PYTHONPATH=$ROOT_DIR
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

poetry run torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $SCRIPT \
    --exp_name baseline_ppg \
    --log_dir "$ROOT_DIR/encodec/ablations/baseline/ppg" &


# # Run distributed training with torch.distributed.run (DDP launcher)
# PYTHONPATH=$ROOT_DIR \
# poetry run python -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node=8 \
#   encodec/trainer/init_config.py \
#   --exp_name baseline_ppg \
#   --log_dir "$ROOT_DIR/encodec/ablations/baseline/ppg"
