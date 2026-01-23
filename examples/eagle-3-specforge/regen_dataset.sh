#!/bin/bash

############################################
# Configuration
############################################
# Base model for data generation
MODEL_NAME="openai/gpt-oss-20b"

# SGLang server settings
SERVER_BASE_PORT=30000
CONCURRENCY=128

# Dataset paths
INPUT_FILE_PATH="./cache/dataset/ultrachat_train.jsonl"
OUTPUT_FILE_PATH="./cache/dataset/ultrachat_train_regen_gpt_oss.jsonl"

# Generation parameters
NUM_SAMPLES=10
MAX_LENGTH=4096
TEMPERATURE=0.8
TOP_P=0.95

# HuggingFace upload settings
UPLOAD_TO_HF=true
HF_DATASET_REPO="baseten-admin/gpt-oss-20b-ultrachat-eagle-training-data"

############################################
# Dataset Regeneration
############################################

# Activate environment


cd model-training-SpecForge

source .venv/bin/activate

python scripts/prepare_data.py --dataset ultrachat

echo "============================================"
echo "🔄 Regenerating Eagle Training Dataset"
echo "Model: $MODEL_NAME"
echo "Input: $INPUT_FILE_PATH"
echo "Output: $OUTPUT_FILE_PATH"
echo "HF Repo: $HF_DATASET_REPO"
echo "============================================"

# Verify environment
echo "Python: $(which python)"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "🔍 Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ No GPUs detected"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE_PATH")"

# Launch SGLang servers (one per GPU)
echo "🚀 Starting $NUM_GPUS SGLang servers..."
SERVER_PIDS=()
SERVER_ADDRESSES=()

for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((SERVER_BASE_PORT + i))
    echo "  Starting server on GPU $i, port $PORT..."
    
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model "$MODEL_NAME" \
        --mem-fraction-static 0.75 \
        --cuda-graph-max-bs 128 \
        --tp 1 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $PORT \
        --dtype bfloat16 &
    
    SERVER_PIDS+=($!)
    SERVER_ADDRESSES+=("127.0.0.1:$PORT")
    echo "  Server $i PID: ${SERVER_PIDS[$i]}"
done

# Wait for all servers to be ready
echo "⏳ Waiting for all servers to be ready..."
MAX_WAIT=300  # 5 minutes per server
READY_SERVERS=()
READY_ADDRESSES=()

for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((SERVER_BASE_PORT + i))
    SERVER_ELAPSED=0  # Reset timer for each server
    SERVER_READY=false
    
    while [ $SERVER_ELAPSED -lt $MAX_WAIT ]; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "✅ Server $i ready on port $PORT"
            READY_SERVERS+=("${SERVER_PIDS[$i]}")
            READY_ADDRESSES+=("127.0.0.1:$PORT")
            SERVER_READY=true
            break
        fi
        sleep 5
        SERVER_ELAPSED=$((SERVER_ELAPSED + 5))
        echo "  Waiting for server $i... (${SERVER_ELAPSED}s elapsed)"
    done
    
    if [ "$SERVER_READY" = false ]; then
        echo "⚠️  Server $i on port $PORT failed to start, skipping..."
        kill ${SERVER_PIDS[$i]} 2>/dev/null
    fi
done

# Check if we have at least one server ready
if [ ${#READY_ADDRESSES[@]} -eq 0 ]; then
    echo "❌ No servers started successfully"
    exit 1
fi

echo "✅ ${#READY_ADDRESSES[@]} servers are ready!"

# Build server addresses argument from ready servers only
SERVER_ADDRESSES_STR="${READY_ADDRESSES[@]}"
SERVER_PIDS=("${READY_SERVERS[@]}")

# Generate dataset using Eagle data generation script
echo "🎲 Generating training data using $NUM_GPUS servers..."
python scripts/regenerate_train_data.py \
    --model "$MODEL_NAME" \
    --server-address $SERVER_ADDRESSES_STR \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --input-file-path "$INPUT_FILE_PATH" \
    --output-file-path "$OUTPUT_FILE_PATH" \
    --is-gpt-oss \
    --is-reasoning-model

# Kill all servers
echo "🛑 Stopping all SGLang servers..."
for ((i=0; i<NUM_GPUS; i++)); do
    kill ${SERVER_PIDS[$i]} 2>/dev/null
    echo "  Stopped server $i (PID: ${SERVER_PIDS[$i]})"
done
wait 2>/dev/null

echo "✅ Dataset generation completed!"
echo "Dataset saved to: $OUTPUT_FILE_PATH"

# Upload to HuggingFace Hub if enabled
if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_DATASET_REPO" ]; then
    # Check if output file exists and is not empty
    if [ ! -f "$OUTPUT_FILE_PATH" ]; then
        echo "❌ Error: Output file $OUTPUT_FILE_PATH does not exist"
        exit 1
    fi
    
    if [ ! -s "$OUTPUT_FILE_PATH" ]; then
        echo "❌ Error: Output file $OUTPUT_FILE_PATH is empty"
        exit 1
    fi
    
    echo "📤 Uploading dataset to HuggingFace Hub..."
    python -c "
from datasets import load_dataset

# Load generated JSONL dataset
dataset = load_dataset('json', data_files='$OUTPUT_FILE_PATH')

# Upload to Hub
print(f'Pushing to {\"$HF_DATASET_REPO\"}...')
dataset.push_to_hub('$HF_DATASET_REPO', private=True)
print('✅ Upload complete!')
"
    echo "✅ Dataset uploaded to: https://huggingface.co/datasets/$HF_DATASET_REPO"
else
    echo "ℹ️  HuggingFace upload disabled"
fi

echo "============================================"
echo "✅ Dataset regeneration complete!"
echo "============================================"
