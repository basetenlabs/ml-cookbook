#!/usr/bin/env bash
set -euo pipefail

############################################
# Config (8 GPU ONLY)
############################################
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_NAME="ultrachat"

# HF upload settings (optional)
HF_DATASET_REPO="${HF_DATASET_REPO:-baseten-admin/${DATASET_NAME}_regen_all_${MODEL_NAME//\//_}}"
HF_PRIVATE="${HF_PRIVATE:-true}"
HF_BRANCH="${HF_BRANCH:-main}"

############################################
# Export for Python heredocs (HF section)
############################################
export MODEL_NAME DATASET_NAME MAX_TOKENS TEMPERATURE
export HF_DATASET_REPO HF_PRIVATE HF_BRANCH


pip install -q uv
apt update -y
apt install -y git curl
cd model-training-SpecForge

uv venv -p 3.11
source .venv/bin/activate
uv pip install -v . --prerelease=allow
uv pip install torch-c-dlpack-ext
uv pip install vllm
uv pip install datasets huggingface_hub

export LIBRARY_PATH=/opt/conda/lib:
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

bash ./examples/train_configs.sh
