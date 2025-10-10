#!/bin/bash
set -eux

pip install -U wandb

# cloning megatron-LM separately because in swift, while a subprocess 
# is cloning the repo, another starts trying to install it. This ensures 
# when the repo exists when installation is being attempted.
export MS_CACHE_HOME=/app/.cache/modelscope
export MODELSCOPE_CACHE=/app/.cache/modelscope
mkdir -p $MS_CACHE_HOME/hub/_github
cd $MS_CACHE_HOME/hub/_github
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM --branch core_r0.13.0

cd /app/
dataset="zai-org/LongAlign-10k"
model_id="Qwen/Qwen3-30B-A3B-Instruct-2507"

mcore_model_dir="./converted/Qwen3-30B-A3B-Instruct-2507-mcore"

swift export \
    --model ${model_id} \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --use_hf \
    --output_dir ${mcore_model_dir}

echo "Starting training"

ckpt_dir=${BT_RW_CACHE_DIR}/${BT_TRAINING_JOB_NAME}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=$BT_NUM_GPUS NNODES=$BT_GROUP_SIZE NODE_RANK=$BT_NODE_RANK MASTER_ADDR=$BT_LEADER_ADDR megatron sft \
    --load ${mcore_model_dir} \
    --dataset ${dataset} \
    --no_initialization false \
    --split_dataset_ratio 0.01 \
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
    --save ${ckpt_dir} \
    --eval_interval 40 \
    --save_interval 40 \
    --max_length 32000 \
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
    --wandb_exp_name $BT_TRAINING_JOB_NAME 

# Checkpoints will be persisted to the Basten cache. 
# You can then sync these checkpoint files into the checkpointing 
# directory by copying or moving them to the $BT_CHECKPOINT_DIR 
# in a separate job, or move them over after training.
