#!/bin/bash
set -euo pipefail

# Install system dependencies
apt-get update
apt-get install -y build-essential curl fzf ripgrep git git-lfs tmux htop lsof gh

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone KingKong (experimental LLM library built on TorchTitan)
mkdir -p workspace
cd workspace
echo "Cloning KingKong..."
git clone https://github.com/basetenlabs/kingkong
cd kingkong

# Set up Python environment
echo "Creating virtual environment..."
uv venv
. .venv/bin/activate
uv sync
# Nightly PyTorch required for torchtitan
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install safetensors wandb

# Download model assets
python scripts/download_hf_assets.py \
    --repo_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local_dir "$BT_TEAM_CACHE_DIR" \
    --all

# Run training
chmod +x run_train.sh
NGPU=8 CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/nemotron3-nano-30B-finetune.toml" \
    ./run_train.sh \
    --checkpoint.initial_load_path "$BT_TEAM_CACHE_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
    --model.hf_assets_path "$BT_TEAM_CACHE_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
    --checkpoint.interval 100 \
    --lr_scheduler.warmup_steps 10 \
    --training.steps 100 \
    --job.dump_folder "$BT_CHECKPOINT_DIR" \
    --training.dataset "hf://HuggingFaceFW/fineweb" \
    --training.dataset_config_name "sample-10BT" \
    --training.dataset_split "train"