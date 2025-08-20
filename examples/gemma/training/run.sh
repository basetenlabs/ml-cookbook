#!/bin/bash
set -eux

pip3 install ring-flash-attn>=0.1.4 #new, along with ^ ring-flash-attn
pip3 install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@7f6afce"

sed -i 's/grad_scale = 1 \/ lse\.numel()/grad_scale = 1 \/ lse\.numel() if lse\.numel() else 1\.0/' /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/cut_cross_entropy/cce.py

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

huggingface-cli login --token=$HF_TOKEN

axolotl preprocess config.yaml

torchrun --nnodes=$BT_GROUP_SIZE --nproc-per-node=$BT_NUM_GPUS --node-rank=$BT_NODE_RANK --rdzv-backend=c10d --rdzv-id=$BT_TRAINING_JOB_ID --rdzv-endpoint=$BT_LEADER_ADDR:29400  -m axolotl.cli.train config.yaml