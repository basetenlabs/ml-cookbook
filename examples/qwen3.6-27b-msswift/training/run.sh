#!/bin/bash
set -eux

# ---------------------------------------------------------------------------
# Mode switch
# ---------------------------------------------------------------------------
# HYDRATE_ONLY=1 only snapshots Qwen3.6-27B into HF_HOME, then exits.
# Subsequent experiment runs start from a warm cache and skip the ~50GB
# download. Set HYDRATE_ONLY=0 (default) for real training runs.
HYDRATE_ONLY=${HYDRATE_ONLY:-0}

# ---------------------------------------------------------------------------
# Hydrate model cache (idempotent; HF will skip files already on disk).
# ---------------------------------------------------------------------------
# All Python deps (ms-swift 4.1.3, transformers 5.6.2, mcore-bridge 1.2.1,
# megatron-core 0.16.1, flash-linear-attention 0.5.0) ship in the base image,
# so there's nothing to pip-install here — just pre-warm the model.
export HF_HOME=$BT_PROJECT_CACHE_DIR/huggingface
mkdir -p $HF_HOME/hub
echo "Snapshotting Qwen/Qwen3.6-27B into $HF_HOME/hub/"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-27B')
"

if [ "$HYDRATE_ONLY" = "1" ]; then
    echo "HYDRATE_ONLY=1: model snapshotted. Exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVE_FULL_MODEL=false
# Tag the checkpoint dir with the seq length / node count of this experiment
# so parallel runs don't clobber each other inside the shared project cache.
EXP_TAG=${EXP_TAG:-default}
checkpoint_dir="$BT_CHECKPOINT_DIR/qwen3.6-27b-lora-${EXP_TAG}"

# ---------------------------------------------------------------------------
# GatedDeltaNet implementation
# ---------------------------------------------------------------------------
# The image's mcore-bridge 1.2.1 + megatron-core 0.16.1 both support the
# Megatron-native GDN path, which is required for packing and TP-on-GDN.
# Set USE_MCORE_GDN=0 only if you specifically need the transformers GDN
# fallback (e.g. comparing implementations).
export USE_MCORE_GDN=${USE_MCORE_GDN:-1}

# ---------------------------------------------------------------------------
# Run-mode-dependent knobs
# ---------------------------------------------------------------------------
MAX_LENGTH=${MAX_LENGTH:-131072}
TRAIN_ITERS=${TRAIN_ITERS:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-25}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-4}

# Parallelism knobs (overridable per experiment via env)
TP=${TP:-8}
PP=${PP:-1}
EP=${EP:-1}
CP=${CP:-1}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=$BT_NUM_GPUS \
NNODES=$BT_GROUP_SIZE \
NODE_RANK=$BT_NODE_RANK \
MASTER_ADDR=$BT_LEADER_ADDR \
megatron sft \
    --model Qwen/Qwen3.6-27B \
    --use_hf 1 \
    --output_dir $checkpoint_dir \
    --dataset 'zai-org/LongAlign-10k' \
    --save_safetensors true \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --finetune true \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP \
    --expert_model_parallel_size $EP \
    --context_parallel_size $CP \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_aux_loss_coeff 1e-3 \
    --moe_expert_capacity_factor 2 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --packing false \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers $RECOMPUTE_NUM_LAYERS \
    --train_iters $TRAIN_ITERS \
    --logging_steps 1 \
    --eval_steps $EVAL_INTERVAL \
    --save_steps 4 \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_length $MAX_LENGTH \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --padding_free false \
    --merge_lora $SAVE_FULL_MODEL
