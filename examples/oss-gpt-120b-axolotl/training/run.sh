#!/bin/bash
set -eux

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

export RDZV_TIMEOUT="1800"
export RDZV_PORT="29400"

apt-get update
apt-get install --no-install-recommends -y netcat-openbsd

export AXOLOTL_CONFIG_FILE=config.yaml

axolotl preprocess $AXOLOTL_CONFIG_FILE

# Followers wait for leader to open the rendezvous port
if [[ "${BT_NODE_RANK}" != "0" ]]; then
  echo "Waiting for leader ${BT_LEADER_ADDR}:${RDZV_PORT}..."
  timeout 20m bash -c 'until nc -z "$0" "$1"; do sleep 5; done' "${BT_LEADER_ADDR}" "${RDZV_PORT}" || {
    echo "Timed out waiting for leader rendezvous port" >&2
    exit 1
  }
fi

if grep -q "^[[:space:]]*output_dir:" $AXOLOTL_CONFIG_FILE; then
    echo "Replacing output_dir"
    sed -i -e "s|^[[:space:]]*output_dir:.*|output_dir: $BT_CHECKPOINT_DIR|" $AXOLOTL_CONFIG_FILE
else
    # If no output_dir exists, append it
    echo "Adding output dir"
    echo "output_dir: $BT_CHECKPOINT_DIR" >> $AXOLOTL_CONFIG_FILE
fi

axolotl train $AXOLOTL_CONFIG_FILE \
    --launcher torchrun -- --nnodes=$BT_GROUP_SIZE --nproc-per-node=$BT_NUM_GPUS --node-rank=$BT_NODE_RANK \
    --rdzv-backend=c10d --rdzv-id=axolotl-${BT_TRAINING_JOB_ID} --rdzv-endpoint=${BT_LEADER_ADDR}:${RDZV_PORT} \
    --rdzv-conf="join_timeout=${RDZV_TIMEOUT}"