#!/bin/bash

pip install -q uv
apt update -y
apt install -y git curl

git clone https://personal_access_token@github.com/basetenlabs/model-training-SpecForge.git
