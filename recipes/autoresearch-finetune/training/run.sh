#!/bin/bash
set -eux

START_TIME=$(date +%s)

# Fine-tune with MS-Swift/Megatron
# The agent modifies this command directly to tune hyperparameters.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=$BT_NUM_GPUS \
megatron sft \
    --model $MODEL \
    --dataset $DATASET \
    --split_dataset_ratio $EVAL_SPLIT_RATIO \
    --save $BT_CHECKPOINT_DIR/lora-output \
    --load_safetensors true \
    --save_safetensors true \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --no_initialization false \
    --lr 8e-5 \
    --min_lr 8e-6 \
    --lr_warmup_fraction 0.1 \
    --weight_decay 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --train_iters 300 \
    --eval_iters 10 \
    --eval_interval 5 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --max_length 4096 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 2 \
    --optimizer_cpu_offload true \
    --sequence_parallel true \
    --use_precision_aware_optimizer true \
    --no_save_optim true \
    --no_save_rng true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --attention_backend flash \
    --use_hf 1 \
    2>&1 | tee /tmp/megatron_output.log

MEGATRON_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

# ---- DO NOT EDIT BELOW THIS LINE ----
# Parse results from Megatron logs
VAL_LOSS=$(grep -oP 'lm loss value:\s*\K[0-9.]+' /tmp/megatron_output.log | tail -1)
PEAK_VRAM_GIB=$(grep -oP 'memory\(GiB\):\s*\K[0-9.]+' /tmp/megatron_output.log | head -1)
PEAK_VRAM=$(python3 -c "print(int(float('${PEAK_VRAM_GIB:-0}') * 1024))" 2>/dev/null || echo "0")

echo "---"
echo "val_loss:         ${VAL_LOSS:-PARSE_FAILED}"
echo "total_seconds:    ${TOTAL_SECONDS}"
echo "peak_vram_mb:     ${PEAK_VRAM}"
exit $MEGATRON_EXIT_CODE
