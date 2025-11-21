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

if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM --branch core_v0.14.0
fi


cd /app/
dataset="winglian/pirate-ultrachat-10k"
model_id="zai-org/GLM-4.6"

mcore_model_dir="$BT_RW_CACHE_DIR/converted/GLM-4.6-mcore"
dataset_path=./dataset

hf download $dataset --repo-type dataset --local-dir $dataset_path

echo "Starting training"

ckpt_dir=$BT_RW_CACHE_DIR/$BT_TRAINING_JOB_NAME
echo "Checkpoint directory: $ckpt_dir"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=$BT_NUM_GPUS NNODES=$BT_GROUP_SIZE NODE_RANK=$BT_NODE_RANK MASTER_ADDR=$BT_LEADER_ADDR megatron sft \
    --load ${mcore_model_dir} \
    --dataset ${dataset_path} \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --no_initialization true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 8 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 16 \
    --expert_tensor_parallel_size 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --recompute_num_layers 4 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --train_iters 200 \
    --eval_iters 10 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-3 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save ${ckpt_dir} \
    --eval_interval 10 \
    --save_interval 10 \
    --max_length 32768 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --use_hf 1 

# Checkpoints will be persisted to the Basten cache. 
# You can then sync these checkpoint files into the checkpointing 
# directory by copying or moving them to the $BT_CHECKPOINT_DIR 
# in a separate job, or move them over after training.