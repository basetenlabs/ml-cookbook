#!/bin/bash
set -eux

apt update
apt install -y curl fzf ripgrep git git-lfs tmux htop lsof gh neovim

pip install --upgrade pip
# pip install megatron-core==0.13.0

pip list | grep swift

# cd /root/
PYTHONPATH=/usr/local/lib/python3.11/site-packages/ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m torch.distributed.run \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node=8 \
    --master_addr $BT_LEADER_ADDR \
    /usr/local/lib/python3.11/site-packages/swift/cli/_megatron/sft.py \
    --model Qwen/Qwen3-30B-A3B \
    --dataset liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT \
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
    --train_iters 2000 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save /Qwen3-30B-A3B-Base \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
