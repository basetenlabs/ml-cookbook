#!/bin/bash
# Fine-tune NVIDIA Isaac-GR00T on a LeRobot-format dataset (Baseten training job).
set -eux

# Persist Hugging Face downloads (base model ~6GB) in the read-write cache volume.
export HF_HOME="${BT_RW_CACHE_DIR:-$HOME/.cache}/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" || true
fi

# System deps used by GR00T data loading.
apt-get update && apt-get install -y --no-install-recommends git git-lfs ffmpeg
git lfs install

# Isaac-GR00T requires Python 3.10. Use uv to build a 3.10 environment that is
# independent of the base image's Python (3.11). Pin to a release tag for
# reproducibility and match the fine-tune command to that tag's README:
#   N1.5 -> scripts/gr00t_finetune.py
#   N1.7 -> gr00t/experiment/launch_finetune.py
pip install -U uv
if [[ ! -d Isaac-GR00T ]]; then
  git clone https://github.com/NVIDIA/Isaac-GR00T.git
fi
cd Isaac-GR00T
uv venv --python 3.10 /opt/groot-venv
# shellcheck disable=SC1091
source /opt/groot-venv/bin/activate
uv pip install -e .
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# Fine-tune. The example trains on the robotics demo dataset bundled in the repo.
# Replace --dataset-path with your own LeRobot v2 dataset (needs meta/modality.json).
# --no-tune_diffusion_model keeps the fine-tune within a 24GB A10G. On a 40GB+ GPU
# (L40S/H100) drop it for a full fine-tune and raise --batch-size.
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/robot_sim.PickNPlace \
  --num-gpus "${BT_NUM_GPUS:-1}" \
  --output-dir "${BT_CHECKPOINT_DIR:-./outputs}" \
  --no-tune_diffusion_model \
  --max-steps 500 \
  --batch-size 8

echo "GR00T fine-tune complete. Checkpoints in ${BT_CHECKPOINT_DIR:-./outputs}"
