#!/bin/bash
set -euo pipefail

# Install system dependencies
apt-get update
apt-get install -y build-essential curl fzf ripgrep git git-lfs tmux htop lsof gh

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone KingKong (Baseten's experimental LLM training library built ontop TorchTitan)
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
uv pip install transformers hf_transfer

# Download model assets
python scripts/download_hf_assets.py \
    --repo_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local_dir "$BT_TEAM_CACHE_DIR" \
    --all

# Run training
chmod +x run_train.sh

export LOG_RANK=0
export NGPU=$BT_NUM_GPUS

# ============================================================
# Supervised Fine-Tuning (SFT) Launch
# ============================================================
# Key SFT flags:
#   --training.dataset_format "messages"    Use chat/instruction format (only assistant tokens trained)
#   --training.document_packing true        Pack multiple conversations into sequences
#   --training.chat_start_sequence          Marks start of assistant response (for masking)
#   --training.chat_end_sequence            Marks end of assistant response (for masking)
#   --training.datasource "huggingface"     Load from HuggingFace Hub (or "local_jsonl")
#   --training.infinite_dataloader true     Loop dataset infinitely

CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/nemotron3-nano-30B-sft.toml" ./run_train.sh \
    --model.hf_assets_path "$BT_TEAM_CACHE_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
    --checkpoint.initial_load_path "$BT_TEAM_CACHE_DIR/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
    --checkpoint.initial_load_in_hf \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval 100 \
    --job.dump_folder "$BT_CHECKPOINT_DIR" \
    --training.dataset "hf://HuggingFaceH4/ultrachat_200k" \
    --training.dataset_split "train_sft" \
    --training.dataset_format "messages" \
    --training.document_packing \
    --training.local_batch_size 1 \
    --training.seq_len 4096 \
    --training.max_norm 1.0 \
    --training.steps 1000 \
    --training.dtype "bfloat16" \
    --optimizer.lr 2e-5 \
    --lr_scheduler.warmup_steps 100