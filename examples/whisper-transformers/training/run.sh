#!/bin/bash
set -eux

pip install -r requirements.txt

# export MODEL_ID="openai/whisper-large-v3"
export MODEL_ID="openai/whisper-large-v3-turbo"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. This script enables --push_to_hub, so an HF token is required to upload checkpoints to Hugging Face."
  echo "Configure Baseten secret 'hf_access_token' and map it to HF_TOKEN in config.py."
fi

### TODO: Update this with a repo you can write to.
# export HF_WRITE_LOC="baseten-admin/gpt-oss-20b-multilingual-reasoner-e2e"

python3 train.py \
    --model_id=$MODEL_ID \
    --dataset_name=DTU54DL/common-accent \
    --mixed_precision_fp16 \
    --push_to_hub \
    --hub_model_id=baseten-admin/whisper-larger-v3-turbo 