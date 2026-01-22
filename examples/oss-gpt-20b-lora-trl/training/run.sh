#!/bin/bash
set -eux

pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0"
# pip install wandb

export MODEL_ID="openai/gpt-oss-20b"
export DATASET_ID="HuggingFaceH4/Multilingual-Thinking"

python3 train.py