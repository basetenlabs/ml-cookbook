#!/bin/bash
# Misc utils
apt update
apt install -y curl fzf ripgrep git git-lfs tmux htop lsof gh neovim

export TUNNEL_NAME="bt-training-rssh"

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz \
    && tar -xf vscode_cli.tar.gz \
    && cp code /usr/local/bin/

code tunnel user login --provider github


echo "Tunnel $TUNNEL_NAME has been started."
echo "If using vscode app, open the command pallet (cmd+shift+P) and type 'Connect to Tunnel'"
echo "If using vscode via browser, go to vscode.dev and click 'Connect to Tunnel' and select '$TUNNEL_NAME'"
tmux new-session -d -s vscode_session "code tunnel --name $TUNNEL_NAME --accept-server-license-terms"

sleep infinity