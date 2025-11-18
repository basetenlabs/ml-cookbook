#!/bin/bash
set -eu pipefail
# Misc utils
apt update
apt install -y curl fzf ripgrep git git-lfs tmux htop lsof gh neovim

export TUNNEL_NAME="bt-training-rssh-${BT_NODE_RANK}"

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz \
    && tar -xf vscode_cli.tar.gz \
    && cp code /usr/local/bin/

code tunnel user login --provider github

echo "Tunnel $TUNNEL_NAME has been started."
echo "If using vscode app, open the command pallet (cmd+shift+P) and type 'Connect to Tunnel'"
echo "If using vscode via browser, go to vscode.dev and click 'Connect to Tunnel' and select '$TUNNEL_NAME'"
tmux new-session -d -s vscode_session "code tunnel --name $TUNNEL_NAME --accept-server-license-terms"

echo "Running from '$(pwd)'"
SCRIPT_PATH="$(pwd)/log_to_baseten.sh"
echo "Run the following command to log all commands and outputs to Baseten:"
echo "source $SCRIPT_PATH"

# Sets up a back to your workspace if you have the cache enabled 
# mkdir -p ${BT_RW_CACHE_DIR}/${BT_TRAINING_JOB_ID}
# watch -n 60 rsync -avh --delete --exclude='.git' --exclude='__pycache__/' ./ ${BT_RW_CACHE_DIR}/${BT_TRAINING_JOB_ID}/


sleep infinity

