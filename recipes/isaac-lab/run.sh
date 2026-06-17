#!/usr/bin/env bash
set -eux

# Task and training length are overridable from the environment.
TASK="${TASK:-Isaac-Cartpole-v0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-150}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/root/IsaacLab}"

# System libraries Isaac Sim's renderer needs. The CUDA/Vulkan driver itself is
# injected by the container runtime (via NVIDIA_DRIVER_CAPABILITIES=all); these are
# the userspace GL/X11/Vulkan loader libs that the base image does not ship.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  libgl1 libglu1-mesa libegl1 libxt6 libxrandr2 libxinerama1 libxcursor1 \
  libxi6 libsm6 libxext6 libxrender1 libglib2.0-0 libvulkan1 vulkan-tools git

# Isaac Sim 5.1 (matches the base image's Python 3.11 + torch 2.7.0+cu128).
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Isaac Lab + the rsl_rl reinforcement learning framework.
if [ ! -d "${ISAACLAB_DIR}" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git --branch main --depth 1 "${ISAACLAB_DIR}"
fi
cd "${ISAACLAB_DIR}"
./isaaclab.sh --install rsl_rl

# Persist run outputs (rsl_rl writes to ./logs/rsl_rl/<task>/<timestamp>/) by pointing
# the logs directory at the Baseten checkpoint path, so policies + TensorBoard logs sync.
mkdir -p /tmp/checkpoints
rm -rf "${ISAACLAB_DIR}/logs"
ln -s /tmp/checkpoints "${ISAACLAB_DIR}/logs"

# Headless RL training. The first run compiles RTX shaders (several minutes); the
# shader cache lives under ~/.cache, which is on the persistent cache mount.
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task="${TASK}" \
  --headless \
  --max_iterations "${MAX_ITERATIONS}"
