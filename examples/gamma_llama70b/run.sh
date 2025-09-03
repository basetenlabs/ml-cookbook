#!/bin/bash
set -eux

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

export RDZV_TIMEOUT="1800"
export RDZV_PORT="29400"

apt-get update
apt-get install --no-install-recommends -y netcat-openbsd

# axolotl preprocess train.yaml

# echo "done"

# Followers wait for leader to open the rendezvous port
# if [[ "${BT_NODE_RANK}" != "0" ]]; then
#   echo "Waiting for leader ${BT_LEADER_ADDR}:${RDZV_PORT}..."
#   timeout 20m bash -c 'until nc -z "$0" "$1"; do sleep 5; done' "${BT_LEADER_ADDR}" "${RDZV_PORT}" || {
#     echo "Timed out waiting for leader rendezvous port" >&2
#     exit 1
#   }
# fi

huggingface-cli login --token $HF_TOKEN

huggingface-cli download baseten-admin/gamma-20aug2025-claudesonnet-good --repo-type dataset

# Check if BT_CHECKPOINT_DIR environment variable exists and is not empty
if [ -n "$BT_CHECKPOINT_DIR" ]; then
    echo "BT_CHECKPOINT_DIR found: $BT_CHECKPOINT_DIR"
    
    # Check if config.yaml exists
    if [ -f "config.yaml" ]; then
        # Use sed to replace the output_dir line
        sed -i "s|output_dir: ./outputs/out/qlora-llama3-70b|output_dir: $BT_CHECKPOINT_DIR|g" config.yaml
        echo "Updated config.yaml output_dir to: $BT_CHECKPOINT_DIR"
    else
        echo "Error: config.yaml file not found"
        exit 1
    fi
else
    echo "BT_CHECKPOINT_DIR not set or empty, no changes made"
fi

axolotl train train_qlora.yaml \
    --launcher torchrun -- --nnodes=$BT_GROUP_SIZE --nproc-per-node=$BT_NUM_GPUS --node-rank=$BT_NODE_RANK \
    --rdzv-backend=c10d --rdzv-id=axolotl-${BT_TRAINING_JOB_ID} --rdzv-endpoint=${BT_LEADER_ADDR}:${RDZV_PORT} \
    --rdzv-conf="join_timeout=${RDZV_TIMEOUT}"