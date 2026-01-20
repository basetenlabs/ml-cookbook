#!/bin/bash

############################################
# Configuration
############################################
# Base model for data generation
MODEL_NAME="Qwen/Qwen3-4B"

# SGLang server settings
SERVER_BASE_PORT=30000
CONCURRENCY=128

# Dataset paths
INPUT_FILE_PATH="./cache/dataset/ultrachat_train.jsonl"
OUTPUT_FILE_PATH="./cache/dataset/ultrachat_train_regen.jsonl"

# Generation parameters
NUM_SAMPLES=10000
MAX_LENGTH=4096
TEMPERATURE=0.8
TOP_P=0.95

# HuggingFace upload settings
UPLOAD_TO_HF=true
HF_DATASET_REPO="baseten-admin/qwen3-4b-eagle-training-data"

############################################
# Dataset Regeneration
############################################

# Activate environment


cd model-training-SpecForge

python scripts/prepare_data.py --dataset ultrachat --sample-size 10000

source .venv/bin/activate

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
MAX_WAIT=300  # 5 minutes
ELAPSED=0

for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((SERVER_BASE_PORT + i))
    while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo "❌ Server $i failed to start within ${MAX_WAIT}s"
            # Kill all servers
            for pid in "${SERVER_PIDS[@]}"; do
                kill $pid 2>/dev/null
            done
            exit 1
        fi
        echo "  Waiting for server $i... (${ELAPSED}s elapsed)"
    done
    echo "✅ Server $i ready on port $PORT"
done

echo "✅ All $NUM_GPUS servers are ready!"

# Build server addresses argument
SERVER_ADDRESSES_STR="${SERVER_ADDRESSES[@]}"

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
    --num-samples "$NUM_SAMPLES"

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
