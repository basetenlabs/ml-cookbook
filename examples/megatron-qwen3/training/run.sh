# single node
export DATASET="zai-org/LongAlign-10k"
export MODEL_ID="Qwen/Qwen3-30B-A3B"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 megatron sft \
    --model $MODEL_ID \
    --dataset $DATASET \
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
    --global_batch_size 4 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 4 \
    --train_iters 200 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save $BT_CHECKPOINT_DIR \
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
    --use_hf