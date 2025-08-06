#!/bin/bash
set -eux

uv pip install deepspeed

uv run accelerate launch \
    --config_file zero3.yaml \
    sft.py \
    --config sft_full.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --packing true packing_strategy wrapped \
    --run_name 20b-full-eager \
    --attn_implementation kernels-community/vllm-flash-attn3