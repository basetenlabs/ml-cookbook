#!/bin/bash
set -eux

pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" trackio

export MODEL_ID="openai/gpt-oss-20b"
export DATASET_ID="HuggingFaceH4/Multilingual-Thinking"

export HF_WRITE_LOC="baseten-admin/gpt-oss-20b-multilingual-reasoner-e2e"

python3 train.py