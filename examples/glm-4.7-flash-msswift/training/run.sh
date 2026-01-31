#!/bin/bash

export HF_HOME=$BT_RW_CACHE_DIR/huggingface
export NPROC_PER_NODE=$BT_NUM_GPUS
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

SAVE_FULL_MODEL=true
checkpoint_dir="$BT_CHECKPOINT_DIR/glm-4.7-flash-lora-32"

megatron sft \
  --model zai-org/GLM-4.7-Flash \
  --use_hf true \
  --check_model false \
  --load_safetensors true \
  --save_safetensors true \
  --merge_lora true \
  --train_type lora \
  --lora_rank 32 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --save ${checkpoint_dir} \
  --dataset 'HuggingFaceTB/everyday-conversations-llama3.1-2k' \
  --custom_dataset_info ./custom_dataset_info.json \
  --template glm4_7 \
  --max_length 32768 \
  --packing true \
  --tensor_model_parallel_size 4 \
  --expert_model_parallel_size 8 \
  --pipeline_model_parallel_size 1 \
  --sequence_parallel true \
  --micro_batch_size 1 \
  --global_batch_size 8 \
  --recompute_granularity selective \
  --recompute_modules core_attn moe \
  --lr 2e-4 \
  --lr_decay_style constant \
  --train_iters 5 \
  --log_interval 1 \
  --save_interval 200 \
  --eval_interval 200 \
  --no_save_optim true \
  --no_save_rng true \
  --report_to wandb \
  --wandb_project GLM-4.7-Flash-sft