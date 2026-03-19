#!/bin/bash
set -eux

# Source the agent-editable experiment config
source ./experiment.env

# Build optional model_type flag
MODEL_TYPE_FLAG=""
if [ -n "${MODEL_TYPE:-}" ]; then
    MODEL_TYPE_FLAG="--model_type $MODEL_TYPE"
fi

START_TIME=$(date +%s)

# Run LoRA fine-tuning via MS-Swift/Megatron
# Pipe output to tee so we can parse it AND print it live
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=$BT_NUM_GPUS \
megatron sft \
    --model $MODEL \
    $MODEL_TYPE_FLAG \
    --dataset $DATASET \
    --split_dataset_ratio $EVAL_SPLIT_RATIO \
    --save $BT_CHECKPOINT_DIR/lora-output \
    --load_safetensors true \
    --save_safetensors true \
    --train_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules $TARGET_MODULES \
    --no_initialization false \
    --lr $LR \
    --min_lr $MIN_LR \
    --lr_warmup_fraction $LR_WARMUP_FRACTION \
    --weight_decay $WEIGHT_DECAY \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --eval_iters 10 \
    --eval_interval 2 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --max_length $MAX_LENGTH \
    --packing $PACKING \
    --recompute_granularity $RECOMPUTE_GRANULARITY \
    --recompute_method $RECOMPUTE_METHOD \
    --recompute_num_layers $RECOMPUTE_NUM_LAYERS \
    --optimizer_cpu_offload $OPTIMIZER_CPU_OFFLOAD \
    --sequence_parallel $SEQUENCE_PARALLEL \
    --use_precision_aware_optimizer true \
    --no_save_optim true \
    --no_save_rng true \
    --num_workers $NUM_WORKERS \
    --dataset_num_proc $DATASET_NUM_PROC \
    --attention_backend flash \
    --use_hf 1 \
    2>&1 | tee /tmp/megatron_output.log

MEGATRON_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

# Parse final eval loss from Megatron logs
# Megatron prints eval results like: "lm loss value: X.XXXX"
VAL_LOSS=$(grep -oP 'lm loss value:\s*\K[0-9.]+' /tmp/megatron_output.log | tail -1)
# Parse peak memory from Megatron's periodic log: "memory(GiB): 23.16"
PEAK_VRAM_GIB=$(grep -oP 'memory\(GiB\):\s*\K[0-9.]+' /tmp/megatron_output.log | head -1)
PEAK_VRAM=$(python3 -c "print(int(float('${PEAK_VRAM_GIB:-0}') * 1024))" 2>/dev/null || echo "0")

# Print structured results block (parsed by the agent from monitor logs)
echo "---"
echo "val_loss:         ${VAL_LOSS:-0.000000}"
echo "total_seconds:    ${TOTAL_SECONDS}"
echo "peak_vram_mb:     ${PEAK_VRAM}"
echo "train_iters:      ${TRAIN_ITERS}"
echo "lora_rank:        ${LORA_RANK}"

exit $MEGATRON_EXIT_CODE
