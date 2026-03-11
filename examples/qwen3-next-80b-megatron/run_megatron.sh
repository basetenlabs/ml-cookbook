#!/usr/bin/env bash
set -euo pipefail

# Validate HF token
[[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]] || {
  echo "ERROR: HF token required for checkpoint upload"; exit 1
}

# Environment
export HF_HOME="/tmp/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export NCCL_SOCKET_IFNAME="^docker0,lo"
export MASTER_PORT="29500"
mkdir -p "$HF_HOME"

# Install dependencies
pip install -q --upgrade pip
pip install -q "ms-swift[llm]==3.12.5" datasets huggingface_hub "transformers==4.57.1"

# Checkpoint directory
checkpoint_dir="${BT_CHECKPOINT_DIR:-/mnt/ckpts}/qwen80b-instruct-megatron-lora"
mkdir -p "$checkpoint_dir"
printf '{}' > "$checkpoint_dir/args.json"

# Workaround: sync args.json to timestamped subdirs created by ms-swift
(while true; do
  for d in "$checkpoint_dir"/v*-*; do
    [[ -d "$d" && ! -f "$d/args.json" ]] && cp "$checkpoint_dir/args.json" "$d/args.json" 2>/dev/null || true
  done
  sleep 1
done) &
trap "kill $! 2>/dev/null" EXIT

# Run training
echo "Starting training: model=Qwen/Qwen3-Next-80B-A3B-Instruct nodes=${BT_GROUP_SIZE}x${BT_NUM_GPUS}gpu"
train_exit=0
NPROC_PER_NODE="$BT_NUM_GPUS" \
NNODES="$BT_GROUP_SIZE" \
NODE_RANK="$BT_NODE_RANK" \
MASTER_ADDR="$BT_LEADER_ADDR" \
MASTER_PORT="$MASTER_PORT" \
megatron sft \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --model_type qwen3_next \
  --save "$checkpoint_dir" \
  --dataset winglian/pirate-ultrachat-10k \
  --template minimax_m2 \
  --check_model false \
  --load_safetensors true \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --merge_lora false \
  --target_modules all-linear \
  --max_epochs 1 \
  --lr_decay_style constant \
  --clip_grad 1.0 \
  --split_dataset_ratio 0.01 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --context_parallel_size 1 \
  --expert_model_parallel_size 8 \
  --bf16 true \
  --loss_scale default \
  --micro_batch_size 1 \
  --global_batch_size 8 \
  --packing false \
  --cross_entropy_loss_fusion true \
  --recompute_granularity selective \
  --recompute_modules core_attn moe \
  --lr 2e-4 \
  --lr_warmup_fraction 0.05 \
  --min_lr 1e-5 \
  --max_length 16384 \
  --save_interval 5 \
  --log_interval 1 \
  --num_workers 8 \
  --dataset_num_proc 8 \
  --lazy_tokenize true \
  --load_from_cache_file true \
  --no_save_optim true \
  --no_save_rng true \
  --sequence_parallel true \
  --attention_backend flash \
  --overlap_grad_reduce false \
  --overlap_param_gather false \
  --use_distributed_optimizer false \
  --use_hf 1 || train_exit=$?

# Upload checkpoint (node 1 uploads, node 0 waits)
hub_repo="baseten-admin/qwen80b-instruct-megatron-lora"
upload_marker="$checkpoint_dir/.upload_done"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$BT_NODE_RANK" == "1" ]]; then
  python "$script_dir/upload_checkpoint.py" "$checkpoint_dir" "$hub_repo"
  touch "$upload_marker"
elif [[ "$BT_NODE_RANK" == "0" ]]; then
  waited=0
  while [[ ! -f "$upload_marker" && $waited -lt 3600 ]]; do sleep 5; ((waited+=5)); done
  [[ -f "$upload_marker" ]] || { echo "Upload timeout"; exit 1; }
fi

# Exit with training status (success if checkpoints exist despite non-zero exit)
if [[ $train_exit -ne 0 ]]; then
  find "$checkpoint_dir" -name "*.safetensors" -type f | grep -q . && exit 0
  exit $train_exit
fi
