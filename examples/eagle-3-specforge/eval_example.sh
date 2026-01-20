#!/usr/bin/env bash
set -euo pipefail

############################################
# Config (8 GPU ONLY)
############################################
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_NAME="ultrachat"

# Regen settings (per server concurrency)
CONCURRENCY_PER_SERVER="${CONCURRENCY_PER_SERVER:-32}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"

# 8 GPU server settings (fixed)
HOST="127.0.0.1"
PORTS=(30001 30002 30003 30004 30005 30006 30007 30008)

SERVER_ADDRS=(
  "${HOST}:30001"
  "${HOST}:30002"
  "${HOST}:30003"
  "${HOST}:30004"
  "${HOST}:30005"
  "${HOST}:30006"
  "${HOST}:30007"
  "${HOST}:30008"
)

# HF upload settings (optional)
HF_DATASET_REPO="${HF_DATASET_REPO:-baseten-admin/${DATASET_NAME}_regen_all_${MODEL_NAME//\//_}}"
HF_PRIVATE="${HF_PRIVATE:-true}"
HF_BRANCH="${HF_BRANCH:-main}"

############################################
# Export for Python heredocs (HF section)
############################################
export MODEL_NAME DATASET_NAME MAX_TOKENS TEMPERATURE
export HF_DATASET_REPO HF_PRIVATE HF_BRANCH

############################################
# Setup
############################################
echo "============================================"
echo "🚀 Regen FULL DATASET + 8 servers (8 GPUs) + (optional) HF upload"
echo "Model:               $MODEL_NAME"
echo "Dataset:             $DATASET_NAME"
echo "GPUs:                8 (fixed)"
echo "Servers:"
printf "  - %s\n" "${SERVER_ADDRS[@]}"
echo "Concurrency/server:  $CONCURRENCY_PER_SERVER"
echo "Max tokens:          $MAX_TOKENS"
echo "Temperature:         $TEMPERATURE"
echo "HF repo (optional):  $HF_DATASET_REPO"
echo "============================================"

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

############################################
# Paths
############################################
cd benchmarks
python bench_eagle3.py \
    --model Qwen/Qwen3-4B   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path baseten-admin/qwen3-4b-eagle3-ultrachat-ttt-9-regen \
    --port 30000 \
    --config-list 1,0,0,0 1,3,1,4 \
    --benchmark-list mtbench:20 \
    --dtype bfloat16
