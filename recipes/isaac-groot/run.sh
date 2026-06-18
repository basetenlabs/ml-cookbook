#!/bin/bash
# Fine-tune NVIDIA Isaac-GR00T on a LeRobot-format dataset (Baseten training job).
set -eux

# Run Weights & Biases in disabled mode — no W&B key is configured in the job, so
# wandb.init() becomes a no-op (no login/network). Don't set WANDB_DISABLED: GR00T
# hardcodes report_to="wandb", and WANDB_DISABLED would make transformers demand
# the package it just disabled. Metrics still stream to stdout (`truss train logs`).
export WANDB_MODE=disabled

# Persist Hugging Face downloads (base model ~6GB) in the read-write cache volume.
# Note: hf_transfer is intentionally NOT enabled — the training pod's egress proxy
# breaks its parallel downloader; the standard downloader respects the proxy.
export HF_HOME="${BT_RW_CACHE_DIR:-$HOME/.cache}/huggingface"
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" || true
fi

# System deps used by GR00T data loading.
apt-get update && apt-get install -y --no-install-recommends git git-lfs ffmpeg
git lfs install

# Isaac-GR00T requires Python 3.10. Build a 3.10 env with uv (independent of the
# base image's 3.11). We pin the n1.5 release so the finetune entrypoint
# (scripts/gr00t_finetune.py) and the flags below stay stable — main has moved to
# N1.7, which uses gr00t/experiment/launch_finetune.py instead.
pip install -U uv
if [[ ! -d Isaac-GR00T ]]; then
  git clone --branch n1.5-release --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git
fi
cd Isaac-GR00T
uv venv --python 3.10 /opt/groot-venv
# shellcheck disable=SC1091
source /opt/groot-venv/bin/activate
uv pip install --upgrade setuptools
uv pip install -e ".[base]"
# flash-attn builds from source against GR00T's torch; cap parallel jobs so the
# compile fits the instance RAM.
MAX_JOBS=2 uv pip install --no-build-isolation flash-attn==2.7.1.post4

# Fine-tune. The recipe trains on the robotics demo dataset bundled in the repo.
# Replace --dataset-path with your own LeRobot v2 dataset (needs meta/modality.json).
# Full fine-tune of GR00T's projector + diffusion action head (the LLM and vision
# tower stay frozen — GR00T's default). Fits an 80GB H100, and the H100 box's large
# host RAM handles the full 3B-model save. To run on a 24GB GPU (A10G/L4) instead,
# add `--lora-rank 64` and a low `--dataloader-num-workers` to fit the smaller box.
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/robot_sim.PickNPlace \
  --num-gpus "${BT_NUM_GPUS:-1}" \
  --output-dir "${BT_CHECKPOINT_DIR:-./outputs}" \
  --max-steps 100 \
  --batch-size 8

echo "GR00T fine-tune complete. Checkpoints in ${BT_CHECKPOINT_DIR:-./outputs}"
