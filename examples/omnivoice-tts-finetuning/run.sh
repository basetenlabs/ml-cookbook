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

# ===================== Configuration =====================
# Default to LJ Speech (~24h of single-speaker English from a public domain
# reader). Override DATASET_REPO/TEXT_COLUMN to use a different dataset.
DATASET_REPO="${DATASET_REPO:-SeanSleat/lj_speech}"
TEXT_COLUMN="${TEXT_COLUMN:-normalized_text}"
LANGUAGE_ID="${LANGUAGE_ID:-en}"

# Layout matches config/data_config_finetune.json, which points at
# data/finetune/tokens/{train,dev}/data.lst. Keep these in sync if you edit one.
DATA_DIR="data/finetune"
TRAIN_JSONL="${DATA_DIR}/manifests/train.jsonl"
DEV_JSONL="${DATA_DIR}/manifests/dev.jsonl"
TOKEN_DIR="${DATA_DIR}/tokens"
CACHE_DIR="./hf_dataset_cache"

# Audio tokenizer used by the tokenization stage (Higgs Audio v2 codec).
TOKENIZER_PATH="${TOKENIZER_PATH:-eustlb/higgs-audio-v2-tokenizer}"

# On Baseten, $BT_CHECKPOINT_DIR is the only directory whose contents get
# persisted by the CheckpointingConfig (see config.py). Writing checkpoints
# anywhere else leaves them on the job's tmpfs and they're lost when the pod
# tears down. Fall back to ./exp/omnivoice_finetune for local dev.
OUTPUT_DIR="${BT_CHECKPOINT_DIR:-exp/omnivoice_finetune}"

# GPUs. OmniVoice is a Qwen3-0.6B-based model, so a single H100 is plenty for
# fine-tuning. Override GPU_IDS/NUM_GPUS for multi-GPU.
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

# Training / data configs. Swap TRAIN_CONFIG to the SDPA variant if your GPU
# does not support flex_attention.
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_config_finetune.json}"
DATA_CONFIG="${DATA_CONFIG:-config/data_config_finetune.json}"

# Optional size / speed knobs (override via env vars when calling this script).
# MAX_SAMPLES defaults to 800 clips (~1.5h of LJ Speech audio, ~6.6s/clip avg).
# Set MAX_SAMPLES= (empty) to use the full dataset.
MAX_SAMPLES="${MAX_SAMPLES-800}"
DEV_SIZE="${DEV_SIZE:-50}"
MAX_WORKERS="${MAX_WORKERS:-32}"
DATASET_SOURCE="${DATASET_SOURCE:-auto}"  # auto | parquet | audiofolder
NJ_PER_GPU="${NJ_PER_GPU:-3}"             # tokenizer worker processes per GPU

# When run on Baseten, config.py exports INIT_FROM_CHECKPOINT pointing at the
# pre-mounted weights directory (e.g. /app/models/k2-fsa/OmniVoice). Outside
# Baseten this is unset and we fall back to the HF repo id baked into
# TRAIN_CONFIG (init_from_checkpoint: "k2-fsa/OmniVoice"). Other optional
# overrides: STEPS, LEARNING_RATE, BATCH_TOKENS.
# =========================================================

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
# Make the omnivoice package importable when running from a source checkout.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---- Stage 0: prepare OmniVoice JSONL manifests (download + materialize wavs) ----
echo "Stage 0: Preparing dataset from ${DATASET_REPO}..."
PREPARE_ARGS=(
  --dataset_repo "${DATASET_REPO}"
  --train_jsonl "${TRAIN_JSONL}"
  --dev_jsonl "${DEV_JSONL}"
  --cache_dir "${CACHE_DIR}"
  --text_column "${TEXT_COLUMN}"
  --language_id "${LANGUAGE_ID}"
  --dev_size "${DEV_SIZE}"
  --max_workers "${MAX_WORKERS}"
  --source "${DATASET_SOURCE}"
)
if [ -n "${MAX_SAMPLES}" ]; then
  PREPARE_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
python prepare.py "${PREPARE_ARGS[@]}"

# ---- Stage 1: tokenize audio into WebDataset shards (train + dev) ----
echo "Stage 1: Tokenizing audio into WebDataset shards..."
SPLIT_JSONLS=("${TRAIN_JSONL}")
SPLIT_NAMES=("train")
if [ "${DEV_SIZE}" -gt 0 ]; then
  SPLIT_JSONLS+=("${DEV_JSONL}")
  SPLIT_NAMES+=("dev")
fi

for i in "${!SPLIT_JSONLS[@]}"; do
  split_jsonl="${SPLIT_JSONLS[$i]}"
  split="${SPLIT_NAMES[$i]}"
  echo "  Tokenizing ${split} from ${split_jsonl}"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl "${split_jsonl}" \
    --tar_output_pattern "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
    --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --nj_per_gpu "${NJ_PER_GPU}" \
    --shuffle True

  echo "  Done. Manifest written to ${TOKEN_DIR}/${split}/data.lst"
done

# ---- Render the training config (apply optional env overrides) ----
RENDERED_TRAIN_CONFIG="${TRAIN_CONFIG}"
if [ -n "${INIT_FROM_CHECKPOINT:-}" ] || [ -n "${STEPS:-}" ] || \
   [ -n "${LEARNING_RATE:-}" ] || [ -n "${BATCH_TOKENS:-}" ]; then
  RENDERED_TRAIN_CONFIG="config/_train_config.rendered.json"
  python - "${TRAIN_CONFIG}" "${RENDERED_TRAIN_CONFIG}" <<'PY'
import json, os, sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = json.load(f)


def override(key, env, cast):
    val = os.environ.get(env)
    if val not in (None, ""):
        cfg[key] = cast(val)


override("init_from_checkpoint", "INIT_FROM_CHECKPOINT", str)
override("steps", "STEPS", int)
override("learning_rate", "LEARNING_RATE", float)
override("batch_tokens", "BATCH_TOKENS", int)

with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote rendered train config to {dst}")
PY
fi

# OmniVoice's trainer saves the AdamW optimizer state (optimizer.bin, ~2x the
# model size) into every checkpoint via accelerator.save_state. That state is
# only needed to *resume* training, not for inference/deployment, so we strip
# it to keep checkpoints lean. A background sweeper deletes it as checkpoints
# appear (so peak disk stays low during training), plus a final sweep below.
# NOTE: with optimizer.bin removed you cannot resume from these checkpoints
# (resume_from_checkpoint); they remain fully loadable via from_pretrained.
strip_optimizer_state() {
  find "${OUTPUT_DIR}" -type f -name 'optimizer*.bin' -delete 2>/dev/null || true
}

( while true; do strip_optimizer_state; sleep 30; done ) &
SWEEPER_PID=$!
trap 'kill "${SWEEPER_PID}" 2>/dev/null || true' EXIT

# ---- Stage 2: fine-tune ----
echo "Stage 2: Fine-tuning..."
accelerate launch \
  --gpu_ids "${GPU_IDS}" \
  --num_processes "${NUM_GPUS}" \
  -m omnivoice.cli.train \
  --train_config "${RENDERED_TRAIN_CONFIG}" \
  --data_config "${DATA_CONFIG}" \
  --output_dir "${OUTPUT_DIR}"

# Stop the sweeper and do a final pass over the last checkpoint(s).
kill "${SWEEPER_PID}" 2>/dev/null || true
strip_optimizer_state

echo "Done. Checkpoints written under ${OUTPUT_DIR} (optimizer state stripped)"
