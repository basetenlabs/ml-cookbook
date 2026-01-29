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
rm -rf venv-datagen
pip install -q uv
apt update -y
apt install -y git curl wget

uv venv venv-datagen -p 3.11
source venv-datagen/bin/activate
uv pip install -v .
uv pip install torch-c-dlpack-ext
uv pip install vllm
uv pip install datasets huggingface_hub
uv pip install ninja


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
MODEL_NAME="Qwen/Qwen3-4B"

# Dataset (HF dataset repo)
CUSTOM_DATASET=Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b

# Output name
OUTPUT_NAME="test_dataset"

# Ensure venv is active
source venv-datagen/bin/activate

echo "============================================"
echo "🏋️  Starting dataset chat template conversation"
echo "Model: $MODEL_NAME"
echo "Dataset: $CUSTOM_DATASET"
echo "Output: $OUTPUT_NAME"
echo "============================================"


cd scripts

python gen_raw_prompt_output_data.py \
    --model-name "$MODEL_NAME" \
    --custom-dataset "$CUSTOM_DATASET" \
    --prompt-column-name "input" \
    --output-column-name "output" \
    --subset-size 200 \
    --output-name "$OUTPUT_NAME" \
    --max-tokens 1024 \
    --temperature 0.1 \
    --regen-data \
    --batch-size 8 \
    --split "stage1"
