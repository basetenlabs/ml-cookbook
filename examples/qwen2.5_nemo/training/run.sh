#!/bin/bash
set -eux

pip install --upgrade pip
pip install wandb
pip install hf_transfer

cd model
python import_ckpt.py
echo "Done loading model"
python train.py