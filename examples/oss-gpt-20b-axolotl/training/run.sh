#!/bin/bash
set -eux

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

export RDZV_TIMEOUT="1800"
export RDZV_PORT="29400"

torchrun \
    --nnodes=$BT_GROUP_SIZE \
    --nproc-per-node=$BT_NUM_GPUS \
    --node-rank=$BT_NODE_RANK \
    --rdzv-backend=c10d \
    --rdzv-id=axolotl-${BT_TRAINING_JOB_ID} \
    --rdzv-endpoint=${BT_LEADER_ADDR}:${RDZV_PORT} \
    --rdzv-conf="join_timeout=${RDZV_TIMEOUT}" \
    train.py
