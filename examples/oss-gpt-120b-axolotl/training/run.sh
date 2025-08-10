#!/bin/bash
set -eux

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

export RDZV_TIMEOUT="1800"
export RDZV_PORT="29400"

apt-get update
apt-get install --no-install-recommends -y netcat-openbsd

axolotl preprocess config.yaml

# Followers wait for leader to open the rendezvous port
if [[ "${BT_NODE_RANK}" != "0" ]]; then
  echo "Waiting for leader ${BT_LEADER_ADDR}:${RDZV_PORT}..."
  timeout 20m bash -c 'until nc -z "$0" "$1"; do sleep 5; done' "${BT_LEADER_ADDR}" "${RDZV_PORT}" || {
    echo "Timed out waiting for leader rendezvous port" >&2
    exit 1
  }
fi

axolotl train config.yaml \
    --launcher torchrun -- --nnodes=$BT_GROUP_SIZE --nproc-per-node=$BT_NUM_GPUS --node-rank=$BT_NODE_RANK \
    --rdzv-backend=c10d --rdzv-id=axolotl-${BT_TRAINING_JOB_ID} --rdzv-endpoint=${BT_LEADER_ADDR}:${RDZV_PORT} \
    --rdzv-conf="join_timeout=${RDZV_TIMEOUT}"