#!/bin/bash

# Clone repo if it doesn't exist
if [ ! -d "model-training-SpecForge" ]; then
    echo "Cloning model-training-SpecForge..."
    git clone https://github_token@github.com/basetenlabs/model-training-SpecForge.git
fi

cd model-training-SpecForge

# Create venv if it doesn't exist
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
    
    # Install flash-attn
    echo "Installing flash-attn..."
    uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl
else
    source .venv/bin/activate
fi

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#ROOT_DIR=$(dirname $SCRIPT_DIR)
#export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Set up CUDA environment
export CUDA_HOME=/opt/conda
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:$CPATH

set -e

############################################
# Configuration
############################################
# Base model (HF repo)
MODEL_NAME="Qwen/Qwen3-4B"

# Eagle head checkpoint (HF repo)
EAGLE_HEAD_CHECKPOINT="baseten-admin/qwen3-4b-eagle3-orpheus"

# Dataset (HF dataset repo)
CUSTOM_DATASET="baseten-admin/qwen3-4b-eagle-training-data"

# Optional eval dataset (update/remove if your dataset doesn't have this split)
EVAL_DATASET="baseten-admin/qwen3-4b-eagle-training-data"

# Training hyperparameters
BATCH_SIZE=1
LEARNING_RATE=1e-4
NUM_EPOCHS=3
MAX_LENGTH=2048
TTT_LENGTH=7
SAVE_INTERVAL=50
EVAL_INTERVAL=500
LOG_INTERVAL=50

# Backbone initialization (recommended for better convergence)
INIT_BACKBONE_LAYER=15  # Use middle layer of Qwen3-4B

# Output directory
OUTPUT_DIR="./output/qwen3-4b-eagle3-custom"

# HuggingFace Hub upload settings
HF_REPO_ID="baseten-admin/qwen3-4b-eagle3-orpheus"  # Change to your desired repo
UPLOAD_TO_HF=true  # Set to false to disable upload

# Other settings
SEED=0
TP_SIZE=1
ATTENTION_BACKEND="flex_attention"
NUM_GPUS=1

############################################
# Train with extended script
############################################

echo "============================================"
echo "🏋️  Starting Qwen3-4B Eagle3 Training"
echo "Model: $MODEL_NAME"
echo "Eagle Head: $EAGLE_HEAD_CHECKPOINT"
echo "Dataset: $CUSTOM_DATASET"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Verify we're in the venv and flash-attn is available
echo "Python: $(.venv/bin/python --version)"
.venv/bin/python -c "from flash_attn import flash_attn_func; print('✅ flash-attn verified')"

# Use torchrun for multi-GPU training
.venv/bin/torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 scripts/train_eagle3_extended.py \
    --target-model-path "$MODEL_NAME" \
    --eagle-head-hf-checkpoint "$EAGLE_HEAD_CHECKPOINT" \
    --train-data-path "$CUSTOM_DATASET" \
    --draft-model-config /workspace/model-training-SpecForge/configs/qwen3-4b-eagle3-auto.json \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --num-epochs "$NUM_EPOCHS" \
    --max-length "$MAX_LENGTH" \
    --ttt-length "$TTT_LENGTH" \
    --save-interval "$SAVE_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --seed "$SEED" \
    --tp-size "$TP_SIZE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --report-to wandb \
    --cache-dir ./cache \
    --hf-repo-id "$HF_REPO_ID"

echo "✅ Training completed!"
echo "Model saved to: $OUTPUT_DIR"
if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_ID" ]; then
    echo "Model will be uploaded to: https://huggingface.co/$HF_REPO_ID"
fi
