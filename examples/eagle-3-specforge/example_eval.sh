#!/bin/bash

pip install -q uv
apt update -y
apt install -y git curl
cd model-training-SpecForge

# Setup virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    pip install -q uv
    apt update -y
    apt install -y git curl wget
    uv venv -p 3.11
    source .venv/bin/activate
    uv pip install -v .
    uv pip install torch-c-dlpack-ext
    uv pip install vllm
    uv pip install datasets huggingface_hub
    uv pip install ninja
else
    echo "Virtual environment already exists. Activating..."
    source .venv/bin/activate
fi


# Set up CUDA environment
export CUDA_HOME=/opt/conda
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:$CPATH
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Disable hf_transfer to avoid I/O errors
export HF_HUB_ENABLE_HF_TRANSFER=0

set -e
source .venv/bin/activate
python ./benchmarks/bench_eagle3.py \
    --model Qwen/Qwen3-4B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path baseten-admin/qwen3-4b-eagle3 \
    --port 30000 \
    --config-list 1,3,1,3 \
    --benchmark-list mtbench:1 \
    --dtype bfloat16
