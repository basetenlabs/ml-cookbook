#!/bin/bash
set -eux

export SHARED_CHECKPOINT_DIR=${BT_RW_CACHE_DIR}/${BT_TRAINING_JOB_NAME}
mkdir -p ${SHARED_CHECKPOINT_DIR}
sync_checkpoints() {
  rsync -avh --delete --exclude='.git' --exclude='__pycache__/' ${SHARED_CHECKPOINT_DIR}/ ${BT_CHECKPOINT_DIR}/
}
sync_checkpoints_loop() {
  while true; do
    sync_checkpoints
    sleep 60  
  done
}

if [ "$BT_NODE_RANK" = "0" ]; then
  # sync checkpoints from shared directory to checkpoint directory
  sync_checkpoints_loop &
  sync_checkpoints_pid=$!
  ray start --head --port=$RAY_SERVICE_PORT
  python ray/rendezvous.py
  echo "Starting training job on node $BT_NODE_RANK"
  ray job submit --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" -- ./train.sh
  kill $sync_checkpoints_pid 2>/dev/null || true
  # one final sync of checkpoints
  sync_checkpoints

else 
  ray start --address=$BT_LEADER_ADDR:$RAY_SERVICE_PORT
  python ray/wait_to_finish.py
fi
echo "Node $BT_NODE_RANK finished"