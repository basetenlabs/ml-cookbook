#!/bin/bash
set -eux

# Setup virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -q -r requirements.txt

# Configuration
# Default to LJ Speech (~24h of single-speaker English from a public domain
# reader). Override DATASET_REPO/TEXT_COLUMN to use a different dataset.
DATASET_REPO="${DATASET_REPO:-SeanSleat/lj_speech}"
TEXT_COLUMN="${TEXT_COLUMN:-normalized_text}"
TRAIN_JSONL="train.jsonl"
CACHE_DIR="./hf_dataset_cache"
# On Baseten, $BT_CHECKPOINT_DIR is the only directory whose contents get
# persisted by the CheckpointingConfig (see config.py). Writing checkpoints
# anywhere else leaves them on the job's tmpfs and they're lost when the pod
# tears down. Fall back to ./output for local dev.
OUTPUT_DIR="${BT_CHECKPOINT_DIR:-output}"

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
# When run on Baseten, config.py exports INIT_MODEL_PATH pointing at the
# pre-mounted weights directory (e.g. /app/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base).
# Outside Baseten, fall back to the HF repo id so sft_12hz.py's
# resolve_model_path() snapshot_downloads it.
INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"

# Preset: starting point on a single H100 for a ~1.5h training corpus
# (~800 sentence-length clips).
# - eff. batch = BATCH_SIZE * GRAD_ACCUM = 4 * 2 = 8
# - cosine LR schedule with WARMUP_RATIO of total optimizer steps
# - EVAL_SPLIT rows are held out for per-epoch validation loss; set to 0 to
#   disable.
BATCH_SIZE=4
GRAD_ACCUM=2
LR=5e-6
EPOCHS=12
WARMUP_RATIO=0.05
SAVE_EVERY_N_EPOCHS=2
SPEAKER_NAME="${SPEAKER_NAME:-ft_speaker}"
EVAL_SPLIT=40

# Optional size / speed knobs (override via env vars when calling this script)
# MAX_SAMPLES defaults to 800 clips, which is ~1.5h of LJ Speech audio
# (~6.6s/clip avg). Set MAX_SAMPLES= (empty) to use the full dataset.
MAX_SAMPLES="${MAX_SAMPLES-800}"
MAX_WORKERS="${MAX_WORKERS:-32}"
DATASET_SOURCE="${DATASET_SOURCE:-auto}"  # auto | parquet | audiofolder

# Use the Rust-based hf_transfer downloader (per-file multipart parallelism).
# `hf-transfer` is in requirements.txt; prepare.py also defaults this on,
# but we set it here too so the env propagates to any other HF calls.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download dataset + precompute audio_codes
echo "Preparing dataset from ${DATASET_REPO}..."
PREPARE_ARGS=(
  --dataset_repo "${DATASET_REPO}"
  --output_jsonl "${TRAIN_JSONL}"
  --cache_dir "${CACHE_DIR}"
  --device "${DEVICE}"
  --tokenizer_model_path "${TOKENIZER_MODEL_PATH}"
  --text_column "${TEXT_COLUMN}"
  --max_workers "${MAX_WORKERS}"
  --source "${DATASET_SOURCE}"
)
if [ -n "${MAX_SAMPLES}" ]; then
  PREPARE_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
python prepare.py "${PREPARE_ARGS[@]}"

# Run fine-tuning
echo "Starting fine-tuning..."
python sft_12hz.py \
  --init_model_path "${INIT_MODEL_PATH}" \
  --output_model_path "${OUTPUT_DIR}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --num_epochs "${EPOCHS}" \
  --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}" \
  --speaker_name "${SPEAKER_NAME}" \
  --eval_split "${EVAL_SPLIT}"
