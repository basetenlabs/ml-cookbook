#!/bin/bash
set -eux

pip install -U wandb msgspec pybind11

# cloning megatron-LM separately because in swift, while a subprocess 
# is cloning the repo, another starts trying to install it. This ensures 
# when the repo exists when installation is being attempted.
export MS_CACHE_HOME=/app/.cache/modelscope
export MODELSCOPE_CACHE=/app/.cache/modelscope
export MEGATRON_LM_PATH=/app/.cache/modelscope/hub/_github/Megatron-LM
mkdir -p $MS_CACHE_HOME/hub/_github
cd $MS_CACHE_HOME/hub/_github
mcore_model_dir="$BT_RW_CACHE_DIR/converted/GLM-4.6-mcore"

if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM --branch core_v0.14.0
fi

if [ -d "$mcore_model_dir" ]; then
    rm -rf $mcore_model_dir
fi

cd /app/

model_id="zai-org/GLM-4.6"

swift export \
    --model ${model_id} \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --use_hf \
    --output_dir ${mcore_model_dir}