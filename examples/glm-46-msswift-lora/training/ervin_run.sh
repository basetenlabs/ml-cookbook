#!/bin/bash
set -eux

# Install deps, including Hugging Face CLI
pip install -U wandb msgspec pybind11 datasets
pip install -U transformers
# Optional: ensure HF token is available if the repos are private
# export HF_TOKEN=hf_your_token_here

# -----------------------------------------------------------------------------
# Clone Megatron-LM
# -----------------------------------------------------------------------------
export MS_CACHE_HOME=/app/.cache/modelscope
export MODELSCOPE_CACHE=/app/.cache/modelscope
mkdir -p $MS_CACHE_HOME/hub/_github
cd $MS_CACHE_HOME/hub/_github
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM --branch core_r0.13.0 || true

# -----------------------------------------------------------------------------
# Download model & dataset from Hugging Face to local directories (using hf download)
# -----------------------------------------------------------------------------
cd /app/

HF_MODEL_REPO="zai-org/GLM-4.6"
HF_DATASET_REPO="zai-org/LongAlign-10k"

MODEL_LOCAL_DIR="/app/models/zai-org/GLM-4.6"
DATASET_LOCAL_DIR="/app/datasets/zai-org/LongAlign-10k"

# Download model
mkdir -p "$(dirname "${MODEL_LOCAL_DIR}")"
hf download "${HF_MODEL_REPO}" --repo-type model --local-dir "${MODEL_LOCAL_DIR}"

# Download dataset
mkdir -p "${DATASET_LOCAL_DIR}"
hf download "${HF_DATASET_REPO}" --repo-type dataset --local-dir "${DATASET_LOCAL_DIR}"

# -----------------------------------------------------------------------------
# Export model to Megatron-Core format
# -----------------------------------------------------------------------------
model_id="${MODEL_LOCAL_DIR}"
mcore_model_dir="${BT_RW_CACHE_DIR}/converted/GLM-4.6-mcore"

export MASTER_PORT=23456

swift export \
    --model "${model_id}" \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir "${mcore_model_dir}"

echo "Starting training"

# -----------------------------------------------------------------------------
# Training config
# -----------------------------------------------------------------------------
ckpt_dir=${BT_RW_CACHE_DIR}/${BT_TRAINING_JOB_NAME}
dataset="${DATASET_LOCAL_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$BT_NUM_GPUS \
NNODES=$BT_GROUP_SIZE \
NODE_RANK=$BT_NODE_RANK \
MASTER_ADDR=$BT_LEADER_ADDR \
megatron sft \
    --load "${mcore_model_dir}" \
    --dataset "${dataset}" \
    --no_initialization false \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 4 \
    --train_iters 200 \
    --eval_iters 40 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save "${ckpt_dir}" \
    --eval_interval 40 \
    --save_interval 40 \
    --max_length 64000 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --use_hf 1 \
    --wandb_project qwen3_moe_megatron \
    --wandb_exp_name "$BT_TRAINING_JOB_NAME"

# Checkpoints will be persisted to the Baseten cache. 
# You can then sync these checkpoint files into the checkpointing 
# directory by copying or moving them to the $BT_CHECKPOINT_DIR 
# in a separate job, or move them over after training.