#!/usr/bin/env bash
set -euo pipefail

pip install -q safetensors

# load_checkpoint_config downloads the Loops sampler adapter under
# /tmp/loaded_checkpoints/. Locate the directory holding adapter_config.json.
ADAPTER_DIR="$(dirname "$(find /tmp/loaded_checkpoints -name adapter_config.json | head -1)")"
echo "Adapter dir: ${ADAPTER_DIR}"

python merge.py \
  --adapter-dir "${ADAPTER_DIR}" \
  --base /app/base \
  --output "${BT_CHECKPOINT_DIR}/merged" \
  --workers 8

echo "Merged checkpoint written to ${BT_CHECKPOINT_DIR}/merged"
