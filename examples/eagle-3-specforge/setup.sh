#!/bin/bash

pip install -q uv
apt update -y
apt install -y git curl wget

git clone https://github_token@github.com/basetenlabs/model-training-SpecForge.git

cd model-training-SpecForge

uv venv -p 3.11
source .venv/bin/activate
uv pip install -v .
uv pip install torch-c-dlpack-ext
uv pip install vllm
uv pip install datasets huggingface_hub
uv pip install ninja  # Speed up compilation

# Set up CUDA environment
export CUDA_HOME=/opt/conda
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:$CPATH

# Install flash-attn from pre-built wheel (much faster and more reliable)
# For PyTorch 2.9, CUDA 12.8, Python 3.11
echo "Installing flash-attn..."

uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl



set -e
