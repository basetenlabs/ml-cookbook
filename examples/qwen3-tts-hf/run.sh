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
OUTPUT_JSONL="train_raw.jsonl"
CACHE_DIR="./hf_dataset_cache"
TRAIN_JSONL="train_with_codes.jsonl"
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

# Preset A: tuned for ~800 sentence-length clips on a single H100.
# - eff. batch = BATCH_SIZE * GRAD_ACCUM = 4 * 2 = 8 (~100 optimizer steps/epoch)
# - cosine LR schedule with WARMUP_RATIO of total optimizer steps
# - more frequent checkpoints so you can A/B by ear
# - EVAL_SPLIT rows are held out for per-epoch validation loss; set to 0 to
#   disable. ~5% of an 800-row corpus is a reasonable starting point.
BATCH_SIZE=4
GRAD_ACCUM=2
LR=5e-6
EPOCHS=12
WARMUP_RATIO=0.05
SAVE_EVERY_N_EPOCHS=2
SPEAKER_NAME="jade"
EVAL_SPLIT=40

# Optional quality / size knobs (override via env vars when calling this script)
MAX_SAMPLES="${MAX_SAMPLES:-}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.0}"
MIN_DURATION="${MIN_DURATION:-0.0}"
MAX_DURATION="${MAX_DURATION:-0.0}"
MAX_WORKERS="${MAX_WORKERS:-32}"
DATASET_SOURCE="${DATASET_SOURCE:-auto}"  # auto | parquet | audiofolder

# Use the Rust-based hf_transfer downloader (per-file multipart parallelism).
# `hf-transfer` is in requirements.txt; download_dataset.py also defaults this
# on, but we set it here too so the env propagates to any other HF calls.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download dataset (baseten-admin/sierra-ft-tts) and convert to JSONL
echo "Downloading sierra-ft-tts dataset..."
DOWNLOAD_ARGS=(
  --output_jsonl "${OUTPUT_JSONL}"
  --cache_dir "${CACHE_DIR}"
  --min_confidence "${MIN_CONFIDENCE}"
  --min_duration "${MIN_DURATION}"
  --max_duration "${MAX_DURATION}"
  --max_workers "${MAX_WORKERS}"
  --source "${DATASET_SOURCE}"
)
if [ -n "${MAX_SAMPLES}" ]; then
  DOWNLOAD_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
python download_dataset.py "${DOWNLOAD_ARGS[@]}"

# Prepare data (extract audio_codes)
echo "Preparing data (extracting audio_codes)..."
python prepare_data.py \
  --device "${DEVICE}" \
  --tokenizer_model_path "${TOKENIZER_MODEL_PATH}" \
  --input_jsonl "${OUTPUT_JSONL}" \
  --output_jsonl "${TRAIN_JSONL}"

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
