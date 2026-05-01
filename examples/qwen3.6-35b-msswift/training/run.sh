#!/bin/bash
set -eux

# ---------------------------------------------------------------------------
# Mode switch
# ---------------------------------------------------------------------------
# HYDRATE_ONLY=1 only (a) installs runtime deps into the project cache and
# (b) snapshots Qwen3.6-35B-A3B into HF_HOME, then exits. Subsequent
# experiment runs start from a warm cache and skip the ~70GB download.
# Set HYDRATE_ONLY=0 (default) for real training runs.
HYDRATE_ONLY=${HYDRATE_ONLY:-0}

# ---------------------------------------------------------------------------
# Dependency upgrades (cached in BT_PROJECT_CACHE_DIR across runs)
# ---------------------------------------------------------------------------
# Qwen3.6-35B-A3B is a hybrid linear-attention MoE: ~30 of its 40 layers are
# GatedDeltaNet (linear attention) and the remaining 10 are full attention.
# That makes flash-linear-attention + causal-conv1d hard requirements, and we
# need ms-swift >= 4.1.0 / transformers == 5.2.* to register the model.
PKG_DIR=$BT_PROJECT_CACHE_DIR/qwen3_6_packages
export PYTHONPATH=$PKG_DIR:${PYTHONPATH:-}

# We use --no-deps to keep pip from yanking torch / megatron-core out from
# under us, but that means we have to pin the auxiliary deps that transformers
# 5.2 actually wants. Notably: huggingface-hub 1.x (transformers 5.2.0 requires
# >=1.3.0,<2.0 and the new 1.x API exposes is_offline_mode again), tokenizers
# 0.22-0.23, safetensors >=0.4.3, accelerate >=1.1, plus peft / liger-kernel
# from the upstream Qwen3.6 best-practice doc.
MISSING="not installed"
SWIFT_OK=$(python -c "import swift; print(swift.__version__)" 2>/dev/null || echo "$MISSING")
TF_OK=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "$MISSING")
HUB_OK=$(python -c "from huggingface_hub import is_offline_mode; print('ok')" 2>/dev/null || echo "$MISSING")
BRIDGE_OK=$(python -c "import importlib.metadata; print(importlib.metadata.version('mcore-bridge'))" 2>/dev/null || echo "$MISSING")
# Short-circuit on "$MISSING" *before* the sort -V gates so non-version
# sentinels never reach the version comparison.
if [ "$SWIFT_OK" = "$MISSING" ] \
   || [ "$(printf '%s\n4.1.0' "$SWIFT_OK" | sort -V | head -1)" != "4.1.0" ] \
   || ! python -c "import transformers; assert transformers.__version__.startswith('5.2.')" 2>/dev/null \
   || [ "$HUB_OK" != "ok" ] \
   || [ "$BRIDGE_OK" = "$MISSING" ]; then
    echo "Upgrading ms-swift / transformers / huggingface_hub / mcore-bridge (swift=$SWIFT_OK tf=$TF_OK hub=$HUB_OK bridge=$BRIDGE_OK)"
    pip install --target=$PKG_DIR --no-deps \
        "ms-swift>=4.1.0" \
        "transformers==5.2.*" \
        "huggingface_hub>=1.3.0,<2.0" \
        "tokenizers>=0.22.0,<=0.23.0" \
        "safetensors>=0.4.3" \
        "accelerate>=1.1.0" \
        "peft>=0.13" \
        "liger-kernel" \
        "qwen_vl_utils>=0.0.14" \
        "mcore-bridge>=1.0.2" \
        "torchao>=0.16" \
        "tilelang" \
        "megatron-core>=0.16"
fi
if ! python -c "from fla.ops.utils import *; import causal_conv1d" 2>/dev/null; then
    # PyPI's latest flash-linear-attention (0.5.0) is missing fla.ops.utils,
    # which transformers 5.2's Qwen3.6 implementation requires. Install from
    # GitHub main (yields fla 0.5.1+).
    echo "Installing flash-linear-attention (git) + causal-conv1d"
    pip install --target=$PKG_DIR --no-deps \
        "git+https://github.com/fla-org/flash-linear-attention.git"
    pip install --target=$PKG_DIR --no-deps --no-build-isolation \
        "git+https://github.com/Dao-AILab/causal-conv1d"
fi

# Hydrate model cache (idempotent; HF will skip files already on disk).
export HF_HOME=$BT_PROJECT_CACHE_DIR/huggingface
mkdir -p $HF_HOME/hub
# Don't pass cache_dir — HF defaults to $HF_HOME/hub/, which is the layout
# huggingface_hub.snapshot_download (and ms-swift's loader) reads at runtime.
echo "Snapshotting Qwen/Qwen3.6-35B-A3B into $HF_HOME/hub/"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-35B-A3B')
"

# In HYDRATE_ONLY mode we stop here — deps and model are now in the project
# cache and subsequent experiment runs will start from a warm state.
if [ "$HYDRATE_ONLY" = "1" ]; then
    echo "HYDRATE_ONLY=1: deps installed, model snapshotted. Exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVE_FULL_MODEL=false
# Tag the checkpoint dir with the seq length / node count of this experiment
# so parallel runs don't clobber each other inside the shared project cache.
EXP_TAG=${EXP_TAG:-default}
checkpoint_dir="$BT_CHECKPOINT_DIR/qwen3.6-35b-a3b-lora-${EXP_TAG}"

# ---------------------------------------------------------------------------
# GatedDeltaNet implementation
# ---------------------------------------------------------------------------
# We install megatron-core 0.16.1 into PKG_DIR (overriding the image's 0.14.1),
# so the Megatron-native GDN path *would* work. We still default to
# USE_MCORE_GDN=0 (transformers fallback) because (a) it's the path actually
# verified end-to-end on H200, (b) packing/TP-on-GDN isn't required for our
# current configs (TP=1, packing=false). Flip to 1 if you want packing or TP>1
# across the GatedDeltaNet sublayers.
export USE_MCORE_GDN=${USE_MCORE_GDN:-0}

# ---------------------------------------------------------------------------
# Run-mode-dependent knobs
# ---------------------------------------------------------------------------
MAX_LENGTH=${MAX_LENGTH:-131072}
TRAIN_ITERS=${TRAIN_ITERS:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-25}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-4}

# Parallelism knobs (overridable per experiment via env)
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
CP=${CP:-1}

# Packing: set to "true" only with USE_MCORE_GDN=1 (Megatron-native GDN can
# handle packed sequence boundaries; transformers fallback can't).
PACKING=${PACKING:-false}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Verified configuration (1 node, 8x H200, 128K seq len, LoRA rank 8):
# - EP=8 shards 256 experts across 8 GPUs (the obvious win).
# - TP=1 because we're on the transformers GDN impl which can't TP the GDN
#   sublayers; with --num-query-groups=2 the dense path is also TP-limited.
# - CP=1: GDN+CP requires further mcore-bridge / Megatron-LM main work that
#   isn't on this image. We compensate with full recompute.
# - recompute_granularity=full / uniform / num_layers=4 keeps activation
#   memory bounded. The 10 full-attention layers dominate at long context.
# - optimizer_cpu_offload + use_precision_aware_optimizer pushes optimizer
#   state to host RAM. Standard for long-context MoE LoRA.
# - --packing false: each LongAlign-10k sample processed at its actual length.
#   At 128K this gave swift-reported peak ~30 GiB, nvidia-smi peak 98 GiB on
#   rank 0 (non-expert tensors), 39-86 GiB on EP ranks; ~40s/iter steady.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=$BT_NUM_GPUS \
NNODES=$BT_GROUP_SIZE \
NODE_RANK=$BT_NODE_RANK \
MASTER_ADDR=$BT_LEADER_ADDR \
megatron sft \
    --model Qwen/Qwen3.6-35B-A3B \
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
    --packing $PACKING \
    --packing_length $MAX_LENGTH \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers $RECOMPUTE_NUM_LAYERS \
    --train_iters $TRAIN_ITERS \
    --logging_steps 1 \
    --eval_steps $EVAL_INTERVAL \
    --save_steps 1000 \
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

# Only check for safetensors on the last node
if [ $BT_NODE_RANK -ne $(($BT_GROUP_SIZE - 1)) ]; then
    sleep infinity
fi

MEGATRON_EXIT_CODE=$?

if [ $MEGATRON_EXIT_CODE -ne 0 ]; then
    if [ -d "$checkpoint_dir" ] && [ -n "$(find "$checkpoint_dir" -name "*.safetensors" -type f 2>/dev/null)" ]; then
        echo "Safetensors found in $checkpoint_dir. Exiting successfully."
        exit 0
    else
        echo "Megatron command failed and no safetensors found in $checkpoint_dir. Exiting with error code."
        exit $MEGATRON_EXIT_CODE
    fi
fi
