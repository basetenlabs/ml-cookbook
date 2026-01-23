#!/bin/bash

pip install -q uv
apt update -y
apt install -y git curl

if [ ! -d "model-training-SpecForge" ]; then
    echo "Cloning model-training-SpecForge..."
    git clone https://token@github.com/basetenlabs/model-training-SpecForge.git
fi

cd model-training-SpecForge

# Always create fresh venv
echo "Setting up virtual environment..."
rm -rf .venv
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



# Set up CUDA environment
export CUDA_HOME=/opt/conda
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:$CPATH

# Disable hf_transfer to avoid I/O errors
export HF_HUB_ENABLE_HF_TRANSFER=0

set -e

############################################
# Configuration
############################################
# Base model (HF repo)
MODEL_NAME="openai/gpt-oss-20b"

# Eagle head checkpoint (HF repo) - set to empty if training from scratch
EAGLE_HEAD_CHECKPOINT=""

# Dataset (HF dataset repo)
CUSTOM_DATASET="baseten-admin/ultrachat-train-regen-gpt-oss"

# Optional eval dataset (update/remove if your dataset doesn't have this split)
EVAL_DATASET="baseten-admin/ultrachat-train-regen-gpt-oss"

# Training hyperparameters
BATCH_SIZE=1
LEARNING_RATE=1e-4
NUM_EPOCHS=3
MAX_LENGTH=4096
TTT_LENGTH=7
SAVE_INTERVAL=500
EVAL_INTERVAL=500
LOG_INTERVAL=50


# Output directory
OUTPUT_DIR=./output/gpt-oss-20b-eagle3-ultrachat-regen

# HuggingFace Hub upload settings
HF_REPO_ID=baseten-admin/gpt-oss-20b-eagle3  # Change to your desired repo
UPLOAD_TO_HF=true  # Set to false to disable upload

# Other settings
SEED=0
TP_SIZE=2  # GPT-OSS-20B requires tensor parallelism
ATTENTION_BACKEND="fa"  # Use flash attention (fa) for draft model
NUM_GPUS=8

############################################
# Train with extended script
############################################

# Ensure venv is active
source .venv/bin/activate

echo "============================================"
echo "🏋️  Starting GPT-OSS-20B Eagle3 Training"
echo "Model: $MODEL_NAME"
echo "Eagle Head: $EAGLE_HEAD_CHECKPOINT"
echo "Dataset: $CUSTOM_DATASET"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Verify we're in the venv and flash-attn is available
echo "Python: $(.venv/bin/python --version)"
.venv/bin/python -c "import flash_attn; print('✅ flash-attn verified')"

# Use torchrun for multi-GPU training
TRAIN_ARGS="--target-model-path $MODEL_NAME \
    --train-data-path $CUSTOM_DATASET \
    --draft-model-config /workspace/model-training-SpecForge/configs/gpt-oss-20B-eagle3.json \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --max-length $MAX_LENGTH \
    --ttt-length $TTT_LENGTH \
    --save-interval $SAVE_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --seed $SEED \
    --tp-size $TP_SIZE \
    --attention-backend $ATTENTION_BACKEND \
    --chat-template gpt-oss-naive \
    --report-to wandb \
    --cache-dir ./cache \
    --hf-repo-id $HF_REPO_ID \
    --target-model-backend sglang \
    --sglang-attention-backend fa3 \
    --dist-timeout 60"

# Add Eagle head checkpoint if specified
if [ -n "$EAGLE_HEAD_CHECKPOINT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --eagle-head-hf-checkpoint \"$EAGLE_HEAD_CHECKPOINT\""
fi

.venv/bin/torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 scripts/train_eagle3_extended.py $TRAIN_ARGS

echo "✅ Training completed!"
echo "Model saved to: $OUTPUT_DIR"
if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_ID" ]; then
    echo "Model will be uploaded to: https://huggingface.co/$HF_REPO_ID"
fi
