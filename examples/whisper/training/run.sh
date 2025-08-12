#!/bin/bash
set -eux

pip install -r requirements.txt

export MODEL_ID="openai/whisper-large-v3"

### TODO: Update this with a repo you can write to.
# export HF_WRITE_LOC="baseten-admin/gpt-oss-20b-multilingual-reasoner-e2e"

python3 train.py \
    --model_name=$MODEL_ID \
    --fp16