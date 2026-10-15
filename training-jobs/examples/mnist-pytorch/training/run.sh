#!/bin/bash
set -eux

pip install -r requirements.txt

python train_mnist.py 