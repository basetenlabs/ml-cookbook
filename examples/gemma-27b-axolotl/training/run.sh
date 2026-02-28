#!/bin/bash
set -eux

pip3 install ring-flash-attn>=0.1.4 #new, along with ^ ring-flash-attn
pip3 install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@318b7e2"
sed -i 's/grad_scale = 1 \/ lse\.numel()/grad_scale = 1 \/ lse\.numel() if lse\.numel() else 1\.0/' /root/miniconda3/envs/py3.11/lib/python3.11/site-packages/cut_cross_entropy/cce.py

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

huggingface-cli login --token=$HF_TOKEN

export AXOLOTL_CONFIG_FILE=config.yaml

axolotl preprocess $AXOLOTL_CONFIG_FILE

# Ensure output_dir is valid YAML even if path has special characters.
python3 - <<'PY'
import os
import yaml

cfg_path = os.environ["AXOLOTL_CONFIG_FILE"]
checkpoint_dir = os.environ["BT_CHECKPOINT_DIR"]

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg["output_dir"] = checkpoint_dir

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Set output_dir to: {checkpoint_dir}")
PY

torchrun --nnodes=$BT_GROUP_SIZE --nproc-per-node=$BT_NUM_GPUS --node-rank=$BT_NODE_RANK --rdzv-backend=c10d --rdzv-id=$BT_TRAINING_JOB_ID --rdzv-endpoint=$BT_LEADER_ADDR:29400  -m axolotl.cli.train $AXOLOTL_CONFIG_FILE