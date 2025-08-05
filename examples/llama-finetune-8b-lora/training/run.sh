#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -eux

# Install dependencies
pip install unsloth

# authenticate with wandb
wandb login $WANDB_API_KEY # defined via Runtime.EnvironmentVariables

python train.py