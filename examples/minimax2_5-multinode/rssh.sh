#!/bin/bash
set -euo pipefail

apt update
apt install -y curl fzf ripgrep git git-lfs tmux htop lsof gh neovim

NODE_RANK="${BT_NODE_RANK:-0}"
TUNNEL_BASE="${TUNNEL_BASE:-rssh-vscode}"
TUNNEL_NAME="${TUNNEL_BASE}-node-${NODE_RANK}"

# keep per-node tunnel state isolated
export VSCODE_CLI_DATA_DIR="/tmp/vscode-cli-${NODE_RANK}"
mkdir -p "$VSCODE_CLI_DATA_DIR"

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
cp code /usr/local/bin/

# Needs auth on each node unless you pre-seed ~/.vscode-cli or token-based login
code tunnel user login --provider github

tmux kill-session -t "vscode_session_${NODE_RANK}" 2>/dev/null || true
tmux new-session -d -s "vscode_session_${NODE_RANK}" \
  "code tunnel --name ${TUNNEL_NAME} --accept-server-license-terms"

echo "Started tunnel: ${TUNNEL_NAME}"
sleep infinity
