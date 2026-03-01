#!/bin/bash
set -eux

pip install trl==0.29.0 peft==0.18.1 datasets==4.6.0

python train.py
