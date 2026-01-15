export HF_HOME=$BT_RW_CACHE_DIR/huggingface

SAVE_FULL_MODEL=false
checkpoint_dir="$BT_CHECKPOINT_DIR/qwen3-235b-a22b-lora-64-128"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=$BT_NUM_GPUS NNODES=$BT_GROUP_SIZE NODE_RANK=$BT_NODE_RANK MASTER_ADDR=$BT_LEADER_ADDR megatron sft \
    --model Qwen/Qwen3-235B-A22B \
    --save $checkpoint_dir \
    --dataset 'winglian/pirate-ultrachat-10k' \
    --load_safetensors true \
    --save_safetensors true \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --no_initialization false \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 16 \
    --recompute_num_layers 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --train_iters 100 \
    --eval_iters 40 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --eval_interval 40 \
    --max_length 16384 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --merge_lora $SAVE_FULL_MODEL \
    --use_hf 1