#!/usr/bin/env bash
set -euo pipefail

############################################
# Config (8 GPU ONLY)
############################################
MODEL_NAME="Qwen/Qwen3-4B"
DATASET_NAME="ultrachat"

# Regen settings (per server concurrency)
CONCURRENCY_PER_SERVER="${CONCURRENCY_PER_SERVER:-32}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"

# 8 GPU server settings (fixed)
HOST="127.0.0.1"
PORTS=(30001 30002 30003 30004 30005 30006 30007 30008)

SERVER_ADDRS=(
  "${HOST}:30001"
  "${HOST}:30002"
  "${HOST}:30003"
  "${HOST}:30004"
  "${HOST}:30005"
  "${HOST}:30006"
  "${HOST}:30007"
  "${HOST}:30008"
)

# HF upload settings (optional)
HF_DATASET_REPO="${HF_DATASET_REPO:-baseten-admin/${DATASET_NAME}_regen_all_${MODEL_NAME//\//_}}"
HF_PRIVATE="${HF_PRIVATE:-true}"
HF_BRANCH="${HF_BRANCH:-main}"

############################################
# Export for Python heredocs (HF section)
############################################
export MODEL_NAME DATASET_NAME MAX_TOKENS TEMPERATURE
export HF_DATASET_REPO HF_PRIVATE HF_BRANCH

############################################
# Setup
############################################
echo "============================================"
echo "🚀 Regen FULL DATASET + 8 servers (8 GPUs) + (optional) HF upload"
echo "Model:               $MODEL_NAME"
echo "Dataset:             $DATASET_NAME"
echo "GPUs:                8 (fixed)"
echo "Servers:"
printf "  - %s\n" "${SERVER_ADDRS[@]}"
echo "Concurrency/server:  $CONCURRENCY_PER_SERVER"
echo "Max tokens:          $MAX_TOKENS"
echo "Temperature:         $TEMPERATURE"
echo "HF repo (optional):  $HF_DATASET_REPO"
echo "============================================"

pip install -q uv
apt update -y
apt install -y git curl

git clone https://githubtoken@github.com/basetenlabs/model-training-SpecForge.git
mv train_configs.sh model-training-SpecForge/examples/train_configs.sh
cd model-training-SpecForge

uv venv -p 3.11
source .venv/bin/activate
uv pip install -v . --prerelease=allow
uv pip install torch-c-dlpack-ext
uv pip install vllm
uv pip install datasets huggingface_hub

python scripts/prepare_data.py --dataset "$DATASET_NAME"

export LIBRARY_PATH=/opt/conda/lib:
export LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

############################################
# Paths
############################################
INPUT_FILE="./cache/dataset/${DATASET_NAME}_train.jsonl"
OUTPUT_FILE="./cache/dataset/${DATASET_NAME}_train_regen.jsonl"

export INPUT_FILE OUTPUT_FILE

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "❌ Input dataset not found: $INPUT_FILE"
  exit 1
fi

############################################
# 1) Launch exactly 8 SGLang servers (one per GPU)
############################################
echo "============================================"
echo "🧠 Launching 8 SGLang servers (GPU 0..7)..."
echo "============================================"

SGLANG_LOG_DIR="sglang_logs"
mkdir -p "$SGLANG_LOG_DIR"

PIDS=()

for gpu in {0..7}; do
  port="${PORTS[$gpu]}"
  addr="${HOST}:${port}"
  echo "➡️  GPU $gpu → $addr"

  CUDA_VISIBLE_DEVICES=$gpu \
  python3 -m sglang.launch_server \
    --model "$MODEL_NAME" \
    --cuda-graph-max-bs 128 \
    --tp 1 \
    --trust-remote-code \
    --host "$HOST" \
    --port "$port" \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    > "${SGLANG_LOG_DIR}/server_gpu${gpu}.log" 2>&1 &

  PIDS+=($!)
done

echo "✅ All 8 servers launched."

############################################
# Cleanup
############################################
cleanup() {
  echo "============================================"
  echo "🧹 Cleaning up SGLang servers..."
  echo "============================================"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

############################################
# 2) Wait for readiness
############################################
echo "============================================"
echo "⏳ Waiting for all servers to be ready..."
echo "============================================"

for gpu in {0..7}; do
  addr="${SERVER_ADDRS[$gpu]}"
  echo "Waiting for GPU $gpu server ($addr)..."

  for t in {1..240}; do
    if curl -s "http://${addr}/v1/models" >/dev/null; then
      echo "✅ GPU $gpu server ready ($addr)"
      break
    fi
    sleep 2
    if [[ $t -eq 240 ]]; then
      echo "❌ Timed out waiting for GPU $gpu server ($addr)"
      echo "---- last 200 lines of log ----"
      tail -n 200 "${SGLANG_LOG_DIR}/server_gpu${gpu}.log" || true
      exit 1
    fi
  done
done

############################################
# 3) Regenerate FULL dataset (EXPLICIT server addresses)
############################################
echo "============================================"
echo "♻️  Regenerating FULL dataset using all 8 servers"
echo "Input: $INPUT_FILE"
echo "Regen output: $OUTPUT_FILE"
echo "============================================"

python scripts/regenerate_train_data.py \
  --model "$MODEL_NAME" \
  --concurrency "$CONCURRENCY_PER_SERVER" \
  --max-tokens "$MAX_TOKENS" \
  --server-address \
    127.0.0.1:30001 \
    127.0.0.1:30002 \
    127.0.0.1:30003 \
    127.0.0.1:30004 \
    127.0.0.1:30005 \
    127.0.0.1:30006 \
    127.0.0.1:30007 \
    127.0.0.1:30008 \
  --temperature "$TEMPERATURE" \
  --input-file-path "$INPUT_FILE" \
  --output-file-path "$OUTPUT_FILE"

echo "✅ Regen completed: $OUTPUT_FILE"

############################################
# 4) Train
############################################
echo "============================================"
echo "🏋️  Starting training..."
echo "IMPORTANT: make sure train_configs.sh points to:"
echo "  $OUTPUT_FILE"
echo "============================================"

bash ./examples/train_configs.sh
