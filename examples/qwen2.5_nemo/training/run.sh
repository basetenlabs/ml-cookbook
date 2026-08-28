#!/bin/bash
set -eux

pip install --upgrade pip
pip install wandb
pip install hf_transfer

export HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
cd model
python import_ckpt.py --model_id=HF_MODEL_ID
echo "Done loading model"
python train.py --model_id=HF_MODEL_ID