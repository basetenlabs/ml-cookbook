# Fine-tune Qwen3-ASR-1.7B with Transformers

This example full-fine-tunes
[`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
on audio/transcript pairs using the Qwen team's
[official SFT recipe](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning).
It adds a reusable Hugging Face dataset-preparation step, Baseten checkpoint
persistence, an H100 preset, evaluation, and a guaranteed `final/` checkpoint.

The default running example prepares 800 utterances from LibriSpeech
`train-clean-100`, holds out 40 for validation, and trains for one epoch.

## Baseten quickstart

Install Truss, configure a Hugging Face access token in the Baseten workspace
as the `hf_access_token` secret, then launch the job:

```bash
pip install -U truss
cd examples/qwen3-asr-transformers
truss train push config.py
```

If that project name already exists in your organization, override it only for
submission:

```bash
TRAINING_PROJECT_NAME="My Qwen3-ASR fine-tune" truss train push config.py
```

The default compute profile in `config.py` is:

| Resource | Value | Why |
| --- | ---: | --- |
| GPU | `1x H100` | 80 GB VRAM for full-parameter BF16 training |
| CPU | `16` cores | Parallel audio loading and preprocessing |
| RAM | `96Gi` | Matches the threshold called out by the upstream FlashAttention installation guidance |
| FlashAttention build jobs | `MAX_JOBS=4` | Prevents an accidental source build from exhausting host RAM |

The model weights are mounted at `/app/models/Qwen/Qwen3-ASR-1.7B`, dataset
artifacts use the Baseten read/write cache, and checkpoints are written under
`$BT_CHECKPOINT_DIR` so they survive job teardown.

## What the recipe does

`run.sh` executes the complete pipeline:

1. Creates an isolated virtual environment and installs Qwen3-ASR plus a
   prebuilt FlashAttention 2 wheel.
2. Runs `prepare.py` to download and resample a Hugging Face audio dataset,
   filter long clips, materialize 16 kHz WAV files, and write train/eval JSONL.
3. Runs the adapted upstream `qwen3_asr_sft.py` trainer in BF16 with
   FlashAttention 2.
4. Saves periodic Trainer checkpoints and a stable, self-contained
   `$BT_CHECKPOINT_DIR/final` checkpoint.

## Dataset requirements

The preparation utility accepts any Hugging Face dataset with:

- an audio column, selected with `AUDIO_COLUMN` (default: `audio`)
- a transcript column, selected with `TEXT_COLUMN` (default: `text`)
- an optional ID column used for filenames (default: `id`)

The prepared manifest follows Qwen3-ASR's native target format:

```jsonl
{"audio":"/absolute/path/0000000-84-121123-0000.wav","text":"language English<asr_text>CHAPTER ONE ..."}
```

If language labels are unavailable, set `LANGUAGE=`. The target will use
`language None<asr_text>...`; as the upstream guide notes, the model then does
not learn language detection from that prefix.

### Dataset controls

| Variable | Default | Description |
| --- | --- | --- |
| `DATASET_REPO` | `openslr/librispeech_asr` | Hugging Face dataset repository |
| `DATASET_CONFIG` | `clean` | Dataset subset/config; empty means none |
| `DATASET_SPLIT` | `train.100` | Source split |
| `AUDIO_COLUMN` | `audio` | Audio feature column |
| `TEXT_COLUMN` | `text` | Transcript column |
| `ID_COLUMN` | `id` | Optional stable sample identifier |
| `LANGUAGE` | `English` | Qwen output-language label |
| `MAX_SAMPLES` | `800` | Maximum accepted examples; empty means all |
| `MAX_DURATION_SECONDS` | `30` | Skip clips above this duration |
| `EVAL_SAMPLES` | `40` | Deterministic validation holdout; `0` disables eval |
| `DATASET_SEED` | `42` | Train/eval split seed |

For another dataset:

```bash
DATASET_REPO=mozilla-foundation/common_voice_17_0 \
DATASET_CONFIG=en \
DATASET_SPLIT=train \
TEXT_COLUMN=sentence \
ID_COLUMN=client_id \
LANGUAGE=English \
MAX_SAMPLES=2000 \
./run.sh
```

For multilingual data, prepare each language separately with its correct Qwen
language name and concatenate the resulting JSONL files before training.

## H100 training preset

| Variable | Default | Description |
| --- | ---: | --- |
| `BATCH_SIZE` | `8` | Per-device micro-batch size |
| `GRAD_ACC` | `16` | Gradient accumulation steps |
| `LR` | `2e-5` | AdamW learning rate from the upstream recipe |
| `EPOCHS` | `1` | Dataset passes |
| `WARMUP_RATIO` | `0.02` | Linear warmup fraction |
| `SAVE_STRATEGY` | `epoch` | Save/evaluate at epoch boundaries |
| `SAVE_TOTAL_LIMIT` | `3` | Maximum periodic checkpoints retained |
| `NUM_WORKERS` | `4` | Trainer data-loader workers |
| `GRADIENT_CHECKPOINTING` | `0` | Set to `1` to trade compute for lower activation memory |

The effective batch size is:

```text
BATCH_SIZE × GRAD_ACC × number_of_GPUs
```

The default is therefore `8 × 16 × 1 = 128`, matching the effective
single-GPU batch of the upstream `batch_size=32, grad_acc=4` example while
using much less peak activation memory.

### Batch size and VRAM

Full fine-tuning has a large fixed cost for model weights, gradients, and
optimizer state. Batch size mainly changes activation memory:

```text
peak VRAM ≈ fixed training state + activations(batch size, padded audio length)
```

Every batch is padded to its longest recording, so one long sample raises the
memory cost of the entire batch. The 30-second preparation limit protects the
default run, but grouping production data into similar-duration buckets is
still recommended.

Useful adjustments:

```bash
# Lower peak VRAM while preserving effective batch 128.
BATCH_SIZE=4 GRAD_ACC=32 ./run.sh

# Lower activation memory further at the cost of extra computation.
BATCH_SIZE=4 GRAD_ACC=32 GRADIENT_CHECKPOINTING=1 ./run.sh

# Try the upstream H100-sized micro-batch after validating clip lengths.
BATCH_SIZE=32 GRAD_ACC=4 ./run.sh
```

Gradient accumulation does not hold all accumulated micro-batches in memory;
it retains gradients while processing each micro-batch sequentially. Ordinary
DDP also does not pool VRAM: each GPU stores a complete model and processes its
own per-device batch.

## Local usage

The same pipeline runs locally on a CUDA machine:

```bash
cd examples/qwen3-asr-transformers
./run.sh
```

For a quick end-to-end smoke test, use the tiny LibriSpeech fixture:

```bash
DATASET_REPO=hf-internal-testing/librispeech_asr_dummy \
DATASET_CONFIG=clean \
DATASET_SPLIT=validation \
MAX_SAMPLES=16 \
EVAL_SAMPLES=2 \
BATCH_SIZE=2 \
GRAD_ACC=2 \
./run.sh
```

To prepare data without starting training:

```bash
python prepare.py \
  --dataset_repo openslr/librispeech_asr \
  --dataset_config clean \
  --dataset_split train.100 \
  --max_samples 800 \
  --eval_samples 40
```

## Checkpoints and resume

Periodic checkpoints are written as `checkpoint-<global_step>`. The trainer
also always writes `final/`, even when a small smoke test does not reach
`SAVE_STEPS`. Each directory includes the processor/tokenizer metadata needed
by `Qwen3ASRModel.from_pretrained`.

To resume locally from the latest periodic checkpoint:

```bash
source .venv/bin/activate
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file train.jsonl \
  --eval_file eval.jsonl \
  --output_dir output \
  --resume 1
```

## Quick inference check

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "output/final",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

result = model.transcribe(audio="path/to/test.wav")
print(result[0].language)
print(result[0].text)
```

## Upstream basis

The training loop retains the upstream Qwen recipe's processor-driven audio
collation, prompt masking, BF16/FP16 selection, AdamW Trainer defaults,
checkpoint metadata copying, and resume behavior. The Baseten adaptation was
based on Qwen3-ASR commit `7c6daf77a2421100f5fb066495372c00129d39ff`.
