#!/bin/bash
set -eux

torchrun \
    --nproc-per-node=$BT_NUM_GPUS \
    train.py
