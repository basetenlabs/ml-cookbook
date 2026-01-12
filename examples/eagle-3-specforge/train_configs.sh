#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
BT_TEAM_CACHE_DIR=${BT_TEAM_CACHE_DIR:? "BT_TEAM_CACHE_DIR env var must be set"}
# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-8}
TP_SIZE=${1:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-4B \
    --draft-model-config $ROOT_DIR/configs/qwen3-4b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_train_regen.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-eagle3-perfectblend-ttt-9-regen \
    --num-epochs 10 \
    --batch-size 1 \
    --ttt-length 9 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --save-interval 50000 \
    --eval-interval 50000 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang
