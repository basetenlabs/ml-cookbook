#!/bin/bash
set -euxo pipefail

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -q -r requirements.txt

# Dataset defaults: a bounded slice of LibriSpeech train-clean-100. Override
# these for another Hugging Face audio dataset.
DATASET_REPO="${DATASET_REPO:-openslr/librispeech_asr}"
DATASET_CONFIG="${DATASET_CONFIG:-clean}"
DATASET_SPLIT="${DATASET_SPLIT:-train.100}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
ID_COLUMN="${ID_COLUMN:-id}"
LANGUAGE="${LANGUAGE:-English}"
MAX_SAMPLES="${MAX_SAMPLES-800}"
MAX_DURATION_SECONDS="${MAX_DURATION_SECONDS:-30}"
EVAL_SAMPLES="${EVAL_SAMPLES:-40}"
DATASET_SEED="${DATASET_SEED:-42}"

CACHE_DIR="${BT_RW_CACHE_DIR:-./cache}/qwen3-asr-dataset"
TRAIN_JSONL="./train.jsonl"
EVAL_JSONL="./eval.jsonl"
OUTPUT_DIR="${BT_CHECKPOINT_DIR:-./output}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
GPU_COUNT="${BT_NUM_GPUS:-${NUM_GPUS:-1}}"

if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU count must be a positive integer; got: ${GPU_COUNT}" >&2
    exit 1
fi

# BATCH_SIZE is the per-device micro-batch. The Baseten config adjusts GRAD_ACC
# with GPU count to preserve an effective batch of 128 without holding 128 audio
# samples in VRAM simultaneously.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOG_STEPS="${LOG_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

export HF_HUB_ENABLE_HF_TRANSFER=1

PREPARE_ARGS=(
  --dataset_repo "${DATASET_REPO}"
  --dataset_split "${DATASET_SPLIT}"
  --audio_column "${AUDIO_COLUMN}"
  --text_column "${TEXT_COLUMN}"
  --id_column "${ID_COLUMN}"
  --language "${LANGUAGE}"
  --cache_dir "${CACHE_DIR}"
  --train_jsonl "${TRAIN_JSONL}"
  --eval_jsonl "${EVAL_JSONL}"
  --eval_samples "${EVAL_SAMPLES}"
  --max_duration_seconds "${MAX_DURATION_SECONDS}"
  --seed "${DATASET_SEED}"
)

if [ -n "${DATASET_CONFIG}" ]; then
  PREPARE_ARGS+=(--dataset_config "${DATASET_CONFIG}")
fi
if [ -n "${MAX_SAMPLES}" ]; then
  PREPARE_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Preparing ${DATASET_REPO} (${DATASET_SPLIT})..."
python prepare.py "${PREPARE_ARGS[@]}"

TRAIN_ARGS=(
  --model_path "${INIT_MODEL_PATH}"
  --train_file "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --grad_acc "${GRAD_ACC}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --warmup_ratio "${WARMUP_RATIO}"
  --save_strategy "${SAVE_STRATEGY}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --log_steps "${LOG_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
)

if [ "${EVAL_SAMPLES}" -gt 0 ]; then
  TRAIN_ARGS+=(--eval_file "${EVAL_JSONL}")
fi

echo "Starting Qwen3-ASR fine-tuning..."
if [ "${GPU_COUNT}" -gt 1 ]; then
    echo "Launching single-node DDP with ${GPU_COUNT} GPUs..."
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc-per-node="${GPU_COUNT}" \
      qwen3_asr_sft.py "${TRAIN_ARGS[@]}"
else
    python qwen3_asr_sft.py "${TRAIN_ARGS[@]}"
fi
