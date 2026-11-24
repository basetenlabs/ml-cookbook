#!/bin/bash
set -eu

set --
if [[ "${WHISPER_SMOKE_TEST:-}" == "1" ]]; then
  echo "Running Whisper smoke test: 2 steps, then evaluate and save."
  set -- --max_steps 2 --warmup_steps 0 --eval_steps 2 --save_steps 2 --logging_steps 1
fi
if [[ -n "${HF_HUB_MODEL_ID:-}" ]]; then
  : "${HF_TOKEN:?Set HF_TOKEN via a Baseten secret before exporting to Hugging Face.}"
  set -- "$@" --push_to_hub --hub_model_id "$HF_HUB_MODEL_ID" --hub_private_repo
fi

pip install -r requirements.txt

python3 train.py \
    --model_id=openai/whisper-large-v3-turbo \
    --dataset_name=DTU54DL/common-accent \
    --mixed_precision_fp16 \
    "$@"
