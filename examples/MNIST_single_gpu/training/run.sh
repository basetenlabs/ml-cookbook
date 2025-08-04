#!/bin/bash
set -eux

pip install -r requirements.txt

export BT_RW_CACHE_DIR="/outputs/mnist"
export BT_CHECKPOINT_DIR="/checkpoints/"

python train_mnist.py 