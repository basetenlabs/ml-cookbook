#!/bin/bash
set -eux
set -eux

export HF_TOKEN=$(cat /secrets/hf_access_token)

pip3 install -r /app/model/requirements.txt

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
#python3 /app/model/download.py
python3 -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /app/model/eagle_head \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 2 \
  --speculative-num-draft-tokens 8 \
  --tp 1 \
  --dtype bfloat16 \
  --port 8000