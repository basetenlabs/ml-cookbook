#!/bin/bash
set -eux

# --- Install prerequisites ---
pip install -U ms-swift wandb

# --- Cache & repo setup ---
export MS_CACHE_HOME=/app/.cache/modelscope
export MODELSCOPE_CACHE=$MS_CACHE_HOME
mkdir -p $MS_CACHE_HOME/hub/_github
cd $MS_CACHE_HOME/hub/_github

# Clone Megatron-LM (prevents race during Swift init)
if [ ! -d "Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM --branch core_r0.13.0
else
  echo "[INFO] Megatron-LM already present."
fi

cd /app/

# --- Dataset & model configuration ---
DATASET="zai-org/LongAlign-10k"
MODEL_ID="Qwen/Qwen3-30B-A3B"
MCORE_MODEL_DIR="./converted/Qwen3-30B-A3B-mcore"

# --- Export ModelScope checkpoint to Megatron-Core format ---
swift export \
    --model "${MODEL_ID}" \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --use_hf \
    --output_dir "${MCORE_MODEL_DIR}"

echo "Model export completed."

# --- Distributed environment setup ---
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NNODES=${BT_GROUP_SIZE:-1}
export NODE_RANK=${BT_NODE_RANK:-0}
export MASTER_ADDR=${BT_LEADER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29400}
export WORLD_SIZE=$NNODES

echo "NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"

# --- Checkpoint save directory ---
CKPT_DIR="${BT_RW_CACHE_DIR:-/app/output}/${BT_TRAINING_JOB_NAME:-qwen3_sft_ckpts}"
mkdir -p "$CKPT_DIR"

# --- Start fine-tuning ---
echo "Starting Megatron-Swift Fine-Tuning"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=${BT_NUM_GPUS:-8} \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
megatron sft \
    --load "${MCORE_MODEL_DIR}" \
    --dataset "${DATASET}" \
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
    --save "${CKPT_DIR}" \
    --eval_interval 40 \
    --save_interval 40 \
    --max_length 16000 \
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
    --wandb_exp_name "${BT_TRAINING_JOB_NAME:-qwen3_sft_run}"

echo "Training completed successfully."
echo "Checkpoints saved under: $CKPT_DIR"
