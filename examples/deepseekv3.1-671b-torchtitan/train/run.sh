#!/bin/bash
set -euo pipefail
# Misc utils
apt update
apt install -y curl fzf ripgrep git git-lfs tmux htop lsof gh neovim
apt-get update && apt-get install -y build-essential
apt-get install nvtop

export TUNNEL_NAME="bt-${BT_TRAINING_JOB_ID}-${BT_NODE_RANK}"

# Configure git credentials if GITHUB_TOKEN is available
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "Configuring git credentials..."
    git config --global credential.helper store
    echo "https://aghilann:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
    chmod 600 ~/.git-credentials
    git config --global url."https://aghilann:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
fi

curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir aghilan-workspace

cd aghilan-workspace
echo "Cloning torchtitan"
git clone https://github.com/basetenlabs/kingkong.git
cd kingkong
uv venv
echo "Creating virtual environment"
. .venv/bin/activate
uv sync
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install wandb safetensors
chmod +x run_train.sh

# =============================================================================
# DeepSeek V3 671B LoRA Fine-Tuning - Multi-Node Training Script
# =============================================================================
# Designed for Baseten multi-node training infrastructure.
#
# Required Baseten Environment Variables:
#   BT_GROUP_SIZE          - Number of nodes in the training group
#   BT_NUM_GPUS            - Number of GPUs per node
#   BT_NODE_RANK           - This node's rank (0 to BT_GROUP_SIZE-1)
#   BT_LEADER_ADDR         - IP address of the leader node (rank 0)
#   BT_TEAM_CACHE_DIR      - Shared cache directory for model weights
#   BT_TRAINING_PROJECT_NAME - WandB project name
#   BT_TRAINING_JOB_NAME   - WandB run/experiment name
#
# Usage:
#   ./scripts/train_deepseek_v3_lora_multinode.sh
# =============================================================================

# rm -rf /root/.cache/team_artifacts/hf_checkpoints

DEBUG="false"  # Set to 'true' for debug (2-layer model), 'false' for production (671B)

# Configuration - Uses Baseten environment variables
NNODES="${BT_GROUP_SIZE:?ERROR: BT_GROUP_SIZE must be set}"
NPROC_PER_NODE="${BT_NUM_GPUS:?ERROR: BT_NUM_GPUS must be set}"
NODE_RANK="${BT_NODE_RANK:?ERROR: BT_NODE_RANK must be set}"
MASTER_ADDR="${BT_LEADER_ADDR:?ERROR: BT_LEADER_ADDR must be set}"
MASTER_PORT="${MASTER_PORT:-29500}"

export NGPU="${BT_NUM_GPUS:?ERROR: BT_NUM_GPUS must be set}"

# Config, assets, and checkpoint paths based on DEBUG flag
if [[ "${DEBUG}" == "true" ]]; then
    CONFIG_FILE="${CONFIG_FILE:-./torchtitan/models/deepseek_v3/train_configs/deepseek_aghilora.toml}"
    HF_ASSETS_PATH="${HF_ASSETS_PATH:-${BT_TEAM_CACHE_DIR}/DeepSeek-v3.1-Base-DEBUG}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BT_TEAM_CACHE_DIR}/hf_checkpoints/dsv3-debug-lora}"
else
    CONFIG_FILE="${CONFIG_FILE:-./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b_lora.toml}"
    HF_ASSETS_PATH="${HF_ASSETS_PATH:-${BT_TEAM_CACHE_DIR}/DeepSeek-V3.1-Base}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BT_TEAM_CACHE_DIR}/hf_checkpoints/dsv3-671b-lora}"
fi

# WandB settings (from Baseten env)
WANDB_PROJECT="${BT_TRAINING_PROJECT_NAME:?ERROR: BT_TRAINING_PROJECT_NAME must be set}"
WANDB_RUN_NAME="${BT_TRAINING_JOB_NAME:?ERROR: BT_TRAINING_JOB_NAME must be set}"

# Environment Setup
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_TIMEOUT=3600000 # 1 hour
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"

# Print Configuration
echo "=============================================="
echo "DeepSeek V3 LoRA Multi-Node Training"
echo "=============================================="
if [[ "${DEBUG}" == "true" ]]; then
    echo "Mode: DEBUG (2-layer model)"
else
    echo "Mode: PRODUCTION (671B)"
fi
echo ""
echo "Baseten Environment:"
echo "  BT_GROUP_SIZE (nodes):     ${NNODES}"
echo "  BT_NUM_GPUS (per node):    ${NPROC_PER_NODE}"
echo "  BT_NODE_RANK:              ${NODE_RANK}"
echo "  BT_LEADER_ADDR:            ${MASTER_ADDR}"
echo "  MASTER_PORT:               ${MASTER_PORT}"
echo ""
echo "Paths:"
echo "  CONFIG_FILE:               ${CONFIG_FILE}"
echo "  HF_ASSETS_PATH:            ${HF_ASSETS_PATH}"
echo "  CHECKPOINT_DIR:            ${CHECKPOINT_DIR}"
echo ""
echo "WandB:"
echo "  BT_TRAINING_PROJECT_NAME:  ${WANDB_PROJECT}"
echo "  BT_TRAINING_JOB_NAME:      ${WANDB_RUN_NAME}"
echo "=============================================="

# Validate Environment
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [[ ! -d "${HF_ASSETS_PATH}" ]]; then
    echo "ERROR: HF assets not found: ${HF_ASSETS_PATH}"
    echo "Please download with: python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.1-Base --local_dir \$BT_TEAM_CACHE_DIR --all"
    exit 1
fi

if [[ "${NODE_RANK}" -lt 0 ]] || [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
    echo "ERROR: NODE_RANK must be between 0 and $((NNODES - 1))"
    exit 1
fi

# Launch Training
echo ""
echo "Starting training on node ${NODE_RANK}..."
echo ""

echo "Checkpoint directory BEFORE training:"
ls -lah "${CHECKPOINT_DIR}" 2>/dev/null || echo "(directory does not exist yet)"
echo ""

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --job.config_file "${CONFIG_FILE}" \
    --job.dump_folder "${CHECKPOINT_DIR}" \
    --model.hf_assets_path "${HF_ASSETS_PATH}" \
    --parallelism.expert_parallel_degree 16 \
    --parallelism.fsdp_reshard_after_forward always \
    --checkpoint.export_dtype bfloat16 \
    --activation_checkpoint.mode full \
    --training.local_batch_size 2 \
    --training.steps 1000 \
    --checkpoint.interval 500 \
    --optimizer.lr 1.5e-4 \
    --lr_scheduler.warmup_steps 0 \
    --lr_scheduler.min_lr_factor 1.0



echo ""
echo "Checkpoint directory AFTER training:"
ls -lah "${CHECKPOINT_DIR}" 2>/dev/null || echo "(directory does not exist)"
echo ""

echo "=============================================="
echo "Training completed on node ${NODE_RANK}"
echo "=============================================="