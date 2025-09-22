#!/bin/bash
set -eux

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

export RDZV_TIMEOUT="1800"
export RDZV_PORT="29400"
# if node rank 0
export AXOLOTL_CONFIG_FILE=config.yaml

axolotl preprocess $AXOLOTL_CONFIG_FILE

# First try to replace existing output_dir line
if grep -q "^[[:space:]]*output_dir:" $AXOLOTL_CONFIG_FILE; then
    sed -i 's/^[[:space:]]*output_dir:.*/output_dir: $BT_CHECKPOINT_DIR/' $AXOLOTL_CONFIG_FILE
else
    # If no output_dir exists, append it
    echo "output_dir: $BT_CHECKPOINT_DIR" >> $AXOLOTL_CONFIG_FILE
fi

axolotl train $AXOLOTL_CONFIG_FILE \
    --launcher torchrun -- --nnodes=$BT_GROUP_SIZE --nproc-per-node=$BT_NUM_GPUS --node-rank=$BT_NODE_RANK \
    --rdzv-backend=c10d --rdzv-id=axolotl-${BT_TRAINING_JOB_ID} --rdzv-endpoint=${BT_LEADER_ADDR}:${RDZV_PORT} \
    --rdzv-conf="join_timeout=${RDZV_TIMEOUT}"