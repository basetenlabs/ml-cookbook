#!/bin/bash
set -euo pipefail

# =============================================================================
# DeepSeek V3 Checkpoint Conversion Script
# =============================================================================
# Converts DCP checkpoints to HuggingFace format and copies to BT_CHECKPOINT_DIR
#
# This script should be run on the leader node after training completes.
# It was extracted from run2.sh to allow manual execution when the conversion
# doesn't complete due to other nodes finishing first.
#
# Required environment variables:
#   BT_TEAM_CACHE_DIR  - Shared cache directory for model weights
#   BT_CHECKPOINT_DIR  - Directory to copy converted checkpoints to
#
# Usage:
#   ./convert.sh
# =============================================================================

# =============================================================================
# Setup torchtitan environment
# =============================================================================
apt update
apt install -y curl git git-lfs

# Configure git credentials if GITHUB_TOKEN is available
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "Configuring git credentials..."
    git config --global credential.helper store
    echo "https://aghilann:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
    chmod 600 ~/.git-credentials
    git config --global url."https://aghilann:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
fi

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

mkdir -p /workspace/aghilan-workspace
cd /workspace/aghilan-workspace

if [[ ! -d "torchtitan" ]]; then
    echo "Cloning torchtitan..."
    git clone https://github.com/aghilann/torchtitan
fi

cd torchtitan
git checkout lora-stuff

if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    uv venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate
uv sync
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install wandb safetensors

# =============================================================================
# Configuration
# =============================================================================

# Required environment variables
BT_TEAM_CACHE_DIR="${BT_TEAM_CACHE_DIR:?ERROR: BT_TEAM_CACHE_DIR must be set}"
BT_CHECKPOINT_DIR="${BT_CHECKPOINT_DIR:?ERROR: BT_CHECKPOINT_DIR must be set}"

# Paths
HF_ASSETS_PATH="${HF_ASSETS_PATH:-${BT_TEAM_CACHE_DIR}/DeepSeek-V3.1-Base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BT_TEAM_CACHE_DIR}/hf_checkpoints/dsv3-671b-lora}"

echo "=============================================="
echo "DeepSeek V3 Checkpoint Conversion"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  BT_TEAM_CACHE_DIR: ${BT_TEAM_CACHE_DIR}"
echo "  BT_CHECKPOINT_DIR: ${BT_CHECKPOINT_DIR}"
echo "  HF_ASSETS_PATH:    ${HF_ASSETS_PATH}"
echo "  CHECKPOINT_DIR:    ${CHECKPOINT_DIR}"
echo "=============================================="
echo ""

# =============================================================================
# Convert Checkpoints to HuggingFace Format
# =============================================================================
echo "=============================================="
echo "Converting Checkpoints to HuggingFace Format"
echo "=============================================="

CHECKPOINT_SUBDIR="${CHECKPOINT_DIR}/checkpoint"
OUTPUT_BASE="${BT_TEAM_CACHE_DIR}/converted_hf_checkpoint"

if [[ -d "${CHECKPOINT_SUBDIR}" ]]; then
    echo "Found checkpoint directories:"
    for dir in "${CHECKPOINT_SUBDIR}"/*/; do
        if [[ -d "$dir" ]]; then
            dir_name=$(basename "$dir")
            echo "  - $dir_name"
        fi
    done
    
    echo ""
    echo "Starting conversions..."
    echo ""
    
    for dir in "${CHECKPOINT_SUBDIR}"/*/; do
        if [[ -d "$dir" ]]; then
            dir_name=$(basename "$dir")
            input_path="${CHECKPOINT_SUBDIR}/${dir_name}"
            output_path="${OUTPUT_BASE}_${dir_name}"
            
            echo "=========================================="
            echo "Converting: $dir_name"
            echo "  Input:  $input_path"
            echo "  Output: $output_path"
            echo "=========================================="
            
            python scripts/checkpoint_conversion/convert_to_hf.py \
                "$input_path" \
                "$output_path" \
                --model_name deepseek_v3 \
                --model_flavor 671B_lora \
                --hf_assets_path "$HF_ASSETS_PATH" \
                --export_dtype bfloat16 \
                --adapters-only \
                --base_model_name_or_path "${BT_TEAM_CACHE_DIR}/DeepSeek-V3.1-Base"
            
            if [ $? -eq 0 ]; then
                echo "✓ Successfully converted $dir_name"
                
                # Copy converted checkpoint to BT_CHECKPOINT_DIR
                dest_path="${BT_CHECKPOINT_DIR}/${dir_name}"
                echo "Copying converted checkpoint to: $dest_path"
                mkdir -p "${BT_CHECKPOINT_DIR}"
                cp -r "$output_path" "$dest_path"
                if [ $? -eq 0 ]; then
                    echo "✓ Successfully copied to $dest_path"
                else
                    echo "✗ Failed to copy to $dest_path"
                fi
            else
                echo "✗ Failed to convert $dir_name"
            fi

        fi
    done
    
    echo "=============================================="
    echo "All checkpoint conversions complete!"
    echo "=============================================="
    echo ""
    echo "Converted checkpoints available at:"
    ls -d "${OUTPUT_BASE}"_* 2>/dev/null || echo "(none found)"
    echo ""
    echo "Checkpoints copied to BT_CHECKPOINT_DIR:"
    ls -la "${BT_CHECKPOINT_DIR}" 2>/dev/null || echo "(none found)"
else
    echo "WARNING: No checkpoint subdirectory found at ${CHECKPOINT_SUBDIR}"
    echo "Skipping conversion."
    exit 1
fi

echo ""
echo "=============================================="
echo "Conversion complete!"
echo "=============================================="
