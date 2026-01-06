#!/usr/bin/env bash
set -euo pipefail

pip install uv
git clone https://github.com/basetenlabs/model-training-SpecForge.git
mv train_configs.sh model-training-SpecForge/examples/train_configs.sh
cd model-training-SpecForge
uv venv -p 3.11
source .venv/bin/activate
uv pip install -v . --prerelease=allow
uv pip install torch-c-dlpack-ext
uv pip install vllm
#python scripts/prepare_data.py --dataset messages --sample-size 10000 --hf-data-path example/hf/path
#torchrun --standalone --nproc_per_node=4 test_gpu.py
#python scripts/prepare_data.py --dataset opc
python scripts/prepare_data.py --dataset perfectblend

export LIBRARY_PATH=/opt/conda/lib:
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

bash ./examples/train_configs.sh