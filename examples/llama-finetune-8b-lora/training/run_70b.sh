#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -eux

# Install dependencies
pip install unsloth

# authenticate with wandb
wandb login $WANDB_API_KEY # defined via Runtime.EnvironmentVariables

export HF_WRITE_LOC="baseten-admin/llama3-8b-ft-lora-e2e"
python train.py \
    --model_id=unsloth/Llama-3.3-70B-Instruct \
    --max_seq_length=8096 \
    --push_to_hub \
    --hub_model_id=baseten-admin/llama70b_nvidia_math