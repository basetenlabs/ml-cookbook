## Fine-Tuning OmniVoice

[OmniVoice](https://github.com/k2-fsa/OmniVoice) is a massively multilingual (600+ languages) zero-shot TTS model built on a diffusion language-model architecture. This recipe fine-tunes it from the pretrained `k2-fsa/OmniVoice` checkpoint on a single-speaker corpus to adapt it toward a specific voice/domain. It uses [LJ Speech](https://huggingface.co/datasets/SeanSleat/lj_speech) (~24 hours of single-speaker English from a public-domain reader, 13.1k clips) as the running example, but any HF dataset that exposes a text column and an audio column will work.

This mirrors the [Qwen3-TTS recipe](../qwen3-tts-transformers/) but uses OmniVoice's own training stack (`omnivoice.scripts.extract_audio_tokens` + `omnivoice.cli.train`) rather than a custom SFT loop, and is adapted from the upstream [`run_finetune.sh`](https://github.com/k2-fsa/OmniVoice/blob/master/examples/run_finetune.sh).

**Note:** The `run.sh` script automatically creates and uses a virtual environment (`.venv`) to avoid conflicts with system Python installations. If running scripts manually, ensure you're using a virtual environment or have the required packages installed.

## Quickstart

To run on Baseten, run the following command to deploy the finetuning job:

```bash
truss train push config.py
```

## Pipeline Overview

Fine-tuning runs in three stages, all wired together by `run.sh`:

| Stage | What it does | Tooling |
| ----- | ------------ | ------- |
| 0 | Download the HF dataset, materialize wav clips, write `train.jsonl` / `dev.jsonl` manifests | `prepare.py` |
| 1 | Tokenize audio into WebDataset shards (`.tar` of codec tokens + `data.lst`) | `omnivoice.scripts.extract_audio_tokens` |
| 2 | Fine-tune from the pretrained checkpoint with `accelerate` | `omnivoice.cli.train` |

Unlike the Qwen3-TTS recipe, audio tokenization is a **separate stage** (Stage 1) rather than being precomputed into the manifest — this matches OmniVoice's upstream pipeline.

## Local Usage Reference

### 0) Prepare the dataset

`prepare.py` downloads a HuggingFace TTS dataset, materializes the wav clips locally, and writes the OmniVoice manifests (`train.jsonl` + `dev.jsonl`).

For LJ Speech (`SeanSleat/lj_speech`), the relevant columns are:

| Column            | Type   | Description                                                         |
| ----------------- | ------ | ------------------------------------------------------------------- |
| `id`              | string | Clip id, e.g. `LJ001-0001`. Used to name the extracted wav files.   |
| `audio`           | audio  | Inline audio bytes (auto-extracted into `{cache_dir}/clips/*.wav`). |
| `text`            | string | Raw transcript ("printed in 1462").                                 |
| `normalized_text` | string | Numbers/dates spelled out ("printed in fourteen sixty-two").        |

For LJ Speech we pass `--text_column normalized_text` because the audio matches the normalized reading.

In general the script needs:

- a text column (any name, picked via `--text_column`, default `text`)
- one of: an `audio` column with embedded bytes (parquet path), or a `file_name` column pointing at a wav in the repo (AudioFolder path)
- optionally an `id` column, used to name the extracted wavs when there is no `file_name`

If your dataset is gated, make sure you're authenticated by setting your `hf_access_token` secret in the Baseten UI (or `HF_TOKEN` locally).

**Download (LJ Speech):**

```bash
python prepare.py \
  --dataset_repo SeanSleat/lj_speech \
  --text_column normalized_text \
  --train_jsonl data/finetune/manifests/train.jsonl \
  --dev_jsonl data/finetune/manifests/dev.jsonl \
  --dev_size 50 \
  --cache_dir ./hf_dataset_cache
```

**Common options:**

- `--dataset_repo` (required): HuggingFace dataset repo id.
- `--text_column`: Transcript column to use (default: `text`; use `normalized_text` for LJ Speech).
- `--train_jsonl` / `--dev_jsonl`: Output manifest paths.
- `--dev_size`: Rows held out (deterministically) for the dev set (default: `50`; set `0` to skip).
- `--max_samples`: Cap the number of clips before the split (default: all). `run.sh` defaults this to `800` (~1.5h of LJ Speech audio).
- `--language_id`: Language code written to every row (default: `en`).
- `--cache_dir`: Local directory for the dataset snapshot (default: `./hf_dataset_cache`).
- `--hf_token`: Override the Hugging Face token used for the download.
- `--max_workers`: Number of concurrent file downloads (default: `32`).
- `--source`: `auto` (default), `parquet`, or `audiofolder`. `parquet` uses HF's auto-converted shards (much faster, embeds audio inline); `audiofolder` downloads each wav separately; `auto` tries parquet first.

### 1) Manifest format

After `prepare.py` runs, each manifest contains one JSON object per line in OmniVoice's expected format:

- `id`: unique sample id (derived from the clip file name)
- `audio_path`: absolute path to the wav clip inside `./hf_dataset_cache/clips/...`
- `text`: transcript corresponding to `audio_path`
- `language_id`: language code (e.g. `en`)

Example:

```jsonl
{"id":"LJ001-0001","audio_path":"/abs/path/clips/LJ001-0001.wav","text":"Printing, in the only sense with which we are at present concerned, ...","language_id":"en"}
{"id":"LJ001-0002","audio_path":"/abs/path/clips/LJ001-0002.wav","text":"in being comparatively modern.","language_id":"en"}
```

`id`, `audio_path`, and `text` are mandatory; `language_id` is optional metadata.

### 2) Tokenize audio into WebDataset shards

Stage 1 converts each manifest into WebDataset shards of precomputed codec tokens using the Higgs Audio v2 tokenizer:

```bash
python -m omnivoice.scripts.extract_audio_tokens \
  --input_jsonl data/finetune/manifests/train.jsonl \
  --tar_output_pattern data/finetune/tokens/train/audios/shard-%06d.tar \
  --jsonl_output_pattern data/finetune/tokens/train/txts/shard-%06d.jsonl \
  --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
  --nj_per_gpu 3 \
  --shuffle True
```

This writes `data/finetune/tokens/train/data.lst`, the manifest of shards referenced by `config/data_config_finetune.json`. Repeat for the `dev` split.

### 3) Fine-tune

Run the end-to-end pipeline (prepare + tokenize + train):

```bash
./run.sh
```

`run.sh` is preset for a single H100 on ~1.5h of audio (~800 LJ Speech clips). Override any setting via env vars (e.g. `STEPS=8000 LEARNING_RATE=5e-6 ./run.sh`).

**Data / dataset knobs:**

| Variable         | Default               | Description                                                                                    |
| ---------------- | --------------------- | ---------------------------------------------------------------------------------------------- |
| `DATASET_REPO`   | `SeanSleat/lj_speech` | HuggingFace dataset repo id passed to `prepare.py`.                                             |
| `TEXT_COLUMN`    | `normalized_text`     | Transcript column. Use `text` for most datasets; LJ Speech uses the normalized variant.        |
| `LANGUAGE_ID`    | `en`                  | Language code stamped on every manifest row.                                                    |
| `MAX_SAMPLES`    | `800`                 | Cap on clips (~1.5h of LJ Speech). Set `MAX_SAMPLES=` (empty) to use the full dataset.          |
| `DEV_SIZE`       | `50`                  | Rows held out for the dev set. Set `0` to skip eval.                                            |
| `MAX_WORKERS`    | `32`                  | Concurrent file downloads in `prepare.py`.                                                      |
| `DATASET_SOURCE` | `auto`                | `auto` \| `parquet` \| `audiofolder` — how `prepare.py` materializes the wavs.                  |
| `NJ_PER_GPU`     | `3`                   | Tokenizer worker processes per GPU in Stage 1.                                                  |

**Training knobs:**

The bulk of the training hyperparameters live in `config/train_config_finetune.json`. The most relevant fields (see [OmniVoice docs/training.md](https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md)):

| Field | Default | Description |
| ----- | ------- | ----------- |
| `init_from_checkpoint` | `k2-fsa/OmniVoice` | Pretrained weights to start from. |
| `steps` | `5000` | Total training steps. Tune to your data/task. |
| `learning_rate` | `1e-5` | Peak LR. Lower than from-scratch training (`1e-4`). |
| `batch_tokens` | `8192` | Token budget per GPU per batch (primary memory control). |
| `attn_implementation` | `flex_attention` | Attention backend. Use the SDPA config if your GPU lacks flex_attention support. |

A few of these can be overridden from `run.sh` via env vars without editing the JSON — `STEPS`, `LEARNING_RATE`, `BATCH_TOKENS`, and `INIT_FROM_CHECKPOINT` (used on Baseten to point at the mounted weights). The rendered config is written to `config/_train_config.rendered.json`.

**Output / GPU knobs:**

| Variable      | Default                          | Description |
| ------------- | -------------------------------- | ----------- |
| `OUTPUT_DIR`  | `${BT_CHECKPOINT_DIR:-exp/omnivoice_finetune}` | Where checkpoints are written. On Baseten, only `$BT_CHECKPOINT_DIR` is persisted. |
| `GPU_IDS`     | `0`                              | GPUs to use, e.g. `0,1,2,3`. |
| `NUM_GPUS`    | `1`                              | Number of processes for `accelerate`. |
| `TRAIN_CONFIG`| `config/train_config_finetune.json` | Swap to `config/train_config_finetune_sdpa.json` for broader GPU compatibility. |

**Common overrides:**

```bash
# Quick smoke test (smaller subset, fewer steps):
MAX_SAMPLES=200 STEPS=500 ./run.sh

# Train on the full LJ Speech corpus:
MAX_SAMPLES= ./run.sh

# SDPA attention (if flex_attention is unsupported on your GPU):
TRAIN_CONFIG=config/train_config_finetune_sdpa.json ./run.sh

# Multi-GPU:
GPU_IDS=0,1,2,3 NUM_GPUS=4 ./run.sh

# Different dataset:
DATASET_REPO=org/my-tts-dataset TEXT_COLUMN=text ./run.sh
```

Checkpoints land under `${OUTPUT_DIR}/checkpoint-<step>` (HF format: `config.json` + `model.safetensors` + tokenizer files), directly loadable with `OmniVoice.from_pretrained(...)`.

> **Optimizer state is stripped.** OmniVoice's trainer also writes the AdamW optimizer state (`optimizer.bin`, ~2× the model size) into each checkpoint for training resumption. `run.sh` deletes these as checkpoints are written (and once more at the end), keeping each checkpoint to its inference-only contents. The trade-off: you can't `resume_from_checkpoint` from a stripped checkpoint. If you need resumable checkpoints, remove the `strip_optimizer_state` sweeper from `run.sh`.

**Monitor with TensorBoard:**

```bash
tensorboard --logdir exp/omnivoice_finetune/tensorboard
```

### 4) Quick inference test

The fine-tuned checkpoint is a drop-in OmniVoice model. Voice cloning with a short reference clip from your target speaker is the most stable mode:

```python
import soundfile as sf
import torch
from omnivoice.models.omnivoice import OmniVoice

model = OmniVoice.from_pretrained(
    "exp/omnivoice_finetune/checkpoint-5000",
    device_map="cuda:0",
    dtype=torch.float16,
)

audio = model.generate(
    text="She said she would be here by noon.",
    ref_audio="hf_dataset_cache/clips/LJ001-0001.wav",
    ref_text="Printing, in the only sense with which we are at present concerned, ...",
)
sf.write("output.wav", audio[0], model.sampling_rate)
```

Or via the CLI that ships with `omnivoice`:

```bash
omnivoice-infer \
  --model exp/omnivoice_finetune/checkpoint-5000 \
  --text "She said she would be here by noon." \
  --ref_audio hf_dataset_cache/clips/LJ001-0001.wav \
  --ref_text "Printing, in the only sense with which we are at present concerned, ..." \
  --output output.wav
```

### 5) Deploy a fine-tuned checkpoint with Truss

The `truss/` folder packages a trained checkpoint as a Baseten deployment. OmniVoice isn't served via vLLM, so this uses a small custom Python model server (`truss/model/model.py`) that loads the checkpoint with `OmniVoice.from_pretrained` and exposes a `predict` endpoint for voice cloning / voice design / auto voice.

**Files:**

- `truss/config.yaml` — Truss deployment spec (image, GPU, checkpoint reference, env vars).
- `truss/model/model.py` — Custom model server wrapping the OmniVoice generate API.
- `truss/call.py` — Minimal Python client that posts text (+ optional reference clip) and writes `output.wav`.

**Point the deployment at your checkpoint**

Edit `truss/config.yaml` and set `training_job_id` to the Baseten training job that produced the checkpoint, plus the relative path you want to serve:

```16:23:examples/omnivoice-tts-finetuning/truss/config.yaml
training_checkpoints:
  download_folder: /models/training_checkpoints
  artifact_references:
    - training_job_id: abc123 # Replace with your training job ID
      paths:
        # OmniVoice writes checkpoint-<step> dirs; pick the step you want.
        - "rank-0/checkpoint-5000"
```

The downloaded artifacts land under `/models/training_checkpoints/<training_job_id>/<path>` inside the container. Update `CHECKPOINT_DIR` in the same file so the model server points at that directory:

```10:14:examples/omnivoice-tts-finetuning/truss/config.yaml
environment_variables:
  HF_HUB_ENABLE_HF_TRANSFER: "true"
  # Path to the fine-tuned checkpoint inside the container. Must match the
  # training_checkpoints path downloaded below. Set to "k2-fsa/OmniVoice" to
  # serve the un-finetuned base model instead.
  CHECKPOINT_DIR: /models/training_checkpoints/abc123/rank-0/checkpoint-5000
```

Replace `abc123` with your `training_job_id` and `checkpoint-5000` with whichever step you listed above.

**Deploy**

From the `examples/omnivoice-tts-finetuning/truss` directory:

```bash
truss push
```

**Call the deployed model**

Once the deployment is live, grab its `MODEL_ID` and `DEPLOYMENT_ID` from the Baseten UI and edit them into `truss/call.py`:

```29:30:examples/omnivoice-tts-finetuning/truss/call.py
MODEL_ID = "..."
DEPLOYMENT_ID = "..."
```

Then run it with your Baseten API key in the environment:

```bash
export BASETEN_API_KEY=...
python truss/call.py
```

`call.py` POSTs the request and writes the returned WAV to `output.wav`.

OmniVoice is a zero-shot voice-clone model — fine-tuning adapts the weights toward your speaker, but the voice at inference is driven by a reference clip rather than a baked-in speaker name (unlike the Qwen3-TTS recipe). So `call.py` does **voice cloning**: it sends a short reference clip from the target speaker plus its transcript on the speech request. By default it pulls one clip (+ transcript) from the LJ Speech dataset via the HF datasets-server, so it runs without a local copy of the training data; set `REF_AUDIO_PATH` / `REF_TEXT` to use your own clip instead.

Request body notes:

- `input` (required): the text to synthesize.
- `ref_audio` / `ref_text`: base64 wav + its transcript for voice cloning (what `call.py` sends). Omitting both falls back to "auto voice", which picks a *random* voice each call.
- `instruct` (optional): style attributes for voice design (e.g. `"female, british accent"`).
- `language` (optional): language name or code (e.g. `English` / `en`).
- generation knobs (optional): `num_step`, `guidance_scale`, `speed`, `duration`.
