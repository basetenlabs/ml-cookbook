#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -eux

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. HF_TOKEN is required to access this model on Hugging Face. Configure Baseten secret 'hf_access_token' and map it to HF_TOKEN in config.py."
  exit 1
fi

# Install dependencies
pip install unsloth

python train.py