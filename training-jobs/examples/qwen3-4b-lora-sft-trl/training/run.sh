#!/bin/bash
set -eux

pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" "datasets"

python train.py
