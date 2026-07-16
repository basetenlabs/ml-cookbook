#!/bin/bash
set -eux

pip install -r requirements.txt

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
export MODEL_ID="nvidia/nemotron-3.5-asr-streaming-0.6b"
export TARGET_LANG="en-US"

# BDN mounts the base checkpoint here (set in config.py). Falls back to a local
# path for dev runs outside Baseten.
INIT_MODEL_MOUNT="${INIT_MODEL_MOUNT:-./assets/$MODEL_ID}"
export DATA_DIR="./data"
NEMO_BRANCH="main"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. The base checkpoint is gated on Hugging Face."
  echo "Configure Baseten secret 'hf_access_token' and map it to HF_TOKEN in config.py."
fi

# ------------------------------------------------------------------
# 1. Locate NeMo example scripts + streaming-prompt fine-tune config
# ------------------------------------------------------------------
# The 26.06 container bundles NeMo source (with all ASR deps already installed),
# so prefer it. Only clone as a fallback for older images / local dev - in which
# case the tree may need deps the base image lacks (e.g. kaldialign).
NEMO_DIR=""
for cand in /opt/NeMo /workspace/NeMo /opt/nemo; do
  if [[ -f "$cand/examples/asr/speech_to_text_finetune.py" ]]; then
    NEMO_DIR="$cand"
    break
  fi
done
if [[ -z "$NEMO_DIR" ]]; then
  NEMO_DIR="/opt/NeMo-src"
  [[ -d "$NEMO_DIR" ]] || git clone -b "$NEMO_BRANCH" https://github.com/NVIDIA-NeMo/NeMo "$NEMO_DIR"
  export PYTHONPATH="$NEMO_DIR:${PYTHONPATH:-}"
  pip install kaldialign
fi
echo "Using NeMo source at: $NEMO_DIR"

# ------------------------------------------------------------------
# 2. Locate the base checkpoint (.nemo)
# ------------------------------------------------------------------
# On Baseten it's already on local disk via BDN (see config.py `weights`), so
# there's no download on billed GPU time. Outside Baseten, pull it once.
if [[ ! -d "$INIT_MODEL_MOUNT" ]]; then
  echo "Checkpoint mount not found; downloading $MODEL_ID (local dev fallback)."
  huggingface-cli download "$MODEL_ID" --local-dir "$INIT_MODEL_MOUNT"
fi
HF_CKPT="$(find "$INIT_MODEL_MOUNT" -name '*.nemo' | head -n 1)"
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
