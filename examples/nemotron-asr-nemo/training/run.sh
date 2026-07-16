#!/bin/bash
set -eux

pip install -r requirements.txt

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
export MODEL_ID="nvidia/nemotron-3.5-asr-streaming-0.6b"
export TARGET_LANG="en-US"

# Where to download the base checkpoint. Prefer the shared team cache so the
# ~2.4GB .nemo file is only pulled once across runs.
ASSETS_DIR="${BT_TEAM_CACHE_DIR:-./assets}"
export DATA_DIR="./data"
NEMO_DIR="/opt/NeMo-src"
# Pin to a tag/commit for reproducibility once one is available for this recipe.
NEMO_BRANCH="main"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. The base checkpoint is gated on Hugging Face."
  echo "Configure Baseten secret 'hf_access_token' and map it to HF_TOKEN in config.py."
fi

# ------------------------------------------------------------------
# 1. NeMo example scripts + streaming-prompt fine-tune config
# ------------------------------------------------------------------
if [[ ! -d "$NEMO_DIR" ]]; then
  git clone -b "$NEMO_BRANCH" https://github.com/NVIDIA-NeMo/NeMo "$NEMO_DIR"
fi
export PYTHONPATH="$NEMO_DIR:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# 2. Download the base checkpoint (.nemo) from Hugging Face
# ------------------------------------------------------------------
CKPT_DIR="$ASSETS_DIR/nemotron-3.5-asr-streaming-0.6b"
huggingface-cli download "$MODEL_ID" --local-dir "$CKPT_DIR"
HF_CKPT="$(find "$CKPT_DIR" -name '*.nemo' | head -n 1)"
echo "Using base checkpoint: $HF_CKPT"

# ------------------------------------------------------------------
# 3. Prepare data (swap prepare_data.py for your own manifest builder)
# ------------------------------------------------------------------
python3 prepare_data.py --data_dir "$DATA_DIR" --target_lang "$TARGET_LANG"

# ------------------------------------------------------------------
# 4. Fine-tune from the base checkpoint
# ------------------------------------------------------------------
# Full fine-tune of the Cache-Aware FastConformer-RNNT streaming model.
# - Prefer a fixed step budget over epochs for streaming/iterable data; AN4 is
#   tiny so we cap with limit_train_batches for a quick smoke run.
# - Increase trainer.max_epochs / point at more data for a real run.
# - Reduce ++model.train_ds.batch_duration if you hit out-of-memory errors.
python3 "$NEMO_DIR/examples/asr/speech_to_text_finetune.py" \
    --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
    --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
    +init_from_nemo_model="$HF_CKPT" \
    ++model.train_ds.manifest_filepath="$DATA_DIR/an4_converted/train_manifest.json" \
    ++model.validation_ds.manifest_filepath="$DATA_DIR/an4_converted/test_manifest.json" \
    ++model.train_ds.batch_duration=200 \
    ++model.optim.name="adamw" \
    ++model.optim.lr=0.1 \
    ++model.optim.weight_decay=0.001 \
    ++model.optim.sched.d_model=1024 \
    ++model.optim.sched.warmup_steps=100 \
    ++trainer.devices="$BT_NUM_GPUS" \
    ++trainer.max_epochs=1 \
    ++trainer.limit_train_batches=60 \
    ++trainer.precision=bf16 \
    ++exp_manager.exp_dir="$BT_CHECKPOINT_DIR" \
    ++exp_manager.use_datetime_version=False \
    ++exp_manager.version=finetune
