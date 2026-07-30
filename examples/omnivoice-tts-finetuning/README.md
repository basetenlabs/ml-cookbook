## Fine-Tuning OmniVoice

[OmniVoice](https://github.com/k2-fsa/OmniVoice) is a massively multilingual (600+ languages) zero-shot TTS model built on a diffusion language-model architecture. This recipe fine-tunes it from the pretrained `k2-fsa/OmniVoice` checkpoint on a single-speaker corpus to adapt it toward a specific voice/domain.

Two data sources are supported:

- **A local utterance-level dataset**: a flat folder of `<id>.wav` clips plus a transcript file with one `<id><TAB><text>` line per clip. Place the folder inside this recipe directory (e.g. `datasets/my_speaker`) and it ships with the training job via `truss train push`.
- **A HuggingFace dataset** (the default): any HF dataset that exposes a text column and an audio column. [LJ Speech](https://huggingface.co/datasets/SeanSleat/lj_speech) (~24 hours of single-speaker English, 13.1k clips) is the running example.

This mirrors the [Qwen3-TTS recipe](../qwen3-tts-transformers/) but uses OmniVoice's own training stack (`omnivoice.scripts.extract_audio_tokens` + `omnivoice.cli.train`) rather than a custom SFT loop, and is adapted from the upstream [`run_finetune.sh`](https://github.com/k2-fsa/OmniVoice/blob/master/examples/run_finetune.sh).

**Note:** The `run.sh` script automatically creates and uses a virtual environment (`.venv`) to avoid conflicts with system Python installations. If running scripts manually, ensure you're using a virtual environment or have the required packages installed.

## Quickstart

To run on Baseten, run the following command to deploy the finetuning job:

```bash
truss train push config.py
```

Everything in this directory — including any local dataset folder you place under `datasets/` — is uploaded with the job (see `.truss_ignore` for exclusions), so a bundled local dataset needs no separate upload step. For datasets much larger than a few GB, prefer uploading to S3/GCS/HF and mounting through a [`WeightsSource`](https://docs.baseten.co/training/your-own-data) instead of bundling.

## Pipeline Overview

Fine-tuning runs in three stages, all wired together by `run.sh`:

| Stage | What it does | Tooling |
| ----- | ------------ | ------- |
| 0 | Parse the local dataset (or download an HF dataset) and write `train.jsonl` / `dev.jsonl` manifests | `prepare.py` |
| 1 | Tokenize audio into WebDataset shards (`.tar` of codec tokens + `data.lst`) | `omnivoice.scripts.extract_audio_tokens` |
| 2 | Fine-tune from the pretrained checkpoint with `accelerate` | `omnivoice.cli.train` |

Unlike the Qwen3-TTS recipe, audio tokenization is a **separate stage** (Stage 1) rather than being precomputed into the manifest — this matches OmniVoice's upstream pipeline.

## Local Usage Reference

### 0) Prepare the dataset

`prepare.py` writes the OmniVoice manifests (`train.jsonl` + `dev.jsonl`) from either a local utterance-level folder or a HuggingFace TTS dataset.

**Local utterance-level dataset (`--local_dir`):**

The expected layout is a flat directory of `<id>.wav` clips plus one transcript file (`*.txt` / `*.tsv`, auto-detected) with one line per clip:

```text
utt_0001	Hey. You up this late again?
utt_0002	That's bad. Do you like to read before sleep?
```

- The delimiter is a tab (`|` also accepted). The file's encoding is auto-detected from its BOM — UTF-8, UTF-8-BOM, and UTF-16 LE/BE all work (UTF-16 with CRLF line endings, common for Windows-exported transcripts, would break a naive UTF-8 reader).
- Transcripts get light normalization: whitespace runs collapse to single spaces and padded quotes (`" quoted title "`) lose their inner padding. Punctuation (em dashes, ellipses) is preserved as prosody cues.
- Clips shorter than `--min_duration` (default 0.5 s) or longer than `--max_duration` (default 30 s) are dropped, as are transcript rows without a wav (and vice versa) — each with a warning.

```bash
python prepare.py \
  --local_dir datasets/my_speaker \
  --train_jsonl data/finetune/manifests/train.jsonl \
  --dev_jsonl data/finetune/manifests/dev.jsonl \
  --dev_size 50
```

**HuggingFace dataset (`--dataset_repo`):**

`prepare.py` downloads the dataset and materializes the wav clips locally. For LJ Speech (`SeanSleat/lj_speech`), the relevant columns are:

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

- `--local_dir` / `--dataset_repo` (exactly one required): local folder vs. HuggingFace dataset repo id.
- `--transcript_file`: transcript path for `--local_dir` (default: the single `*.txt`/`*.tsv` in the folder).
- `--min_duration` / `--max_duration`: duration filter for local clips (defaults: `0.5` / `30` seconds; `--max_duration 0` disables the upper bound).
- `--text_column`: Transcript column for HF datasets (default: `text`; use `normalized_text` for LJ Speech).
- `--train_jsonl` / `--dev_jsonl`: Output manifest paths.
- `--dev_size`: Rows held out (deterministically) for the dev set (default: `50`; set `0` to skip).
- `--max_samples`: Cap the number of clips before the split (default: all). `run.sh` only applies a cap (800) for the LJ Speech demo; local datasets use every clip.
- `--language_id`: Language code written to every row (default: `en`).
- `--cache_dir`: Local directory for the HF dataset snapshot (default: `./hf_dataset_cache`).
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

`run.sh` is preset for a single H100 on a small single-speaker corpus (a few hours of audio). Override any setting via env vars (e.g. `STEPS=3000 LEARNING_RATE=5e-6 ./run.sh`).

**Data / dataset knobs:**

| Variable            | Default                    | Description                                                                                    |
| ------------------- | -------------------------- | ---------------------------------------------------------------------------------------------- |
| `LOCAL_DATASET_DIR` | unset                      | Local utterance-level dataset folder (e.g. `datasets/my_speaker`). Unset -> HF path. |
| `TRANSCRIPT_FILE`   | auto-detected              | Transcript file inside the local dataset folder.                                               |
| `DATASET_REPO`      | `SeanSleat/lj_speech`      | HuggingFace dataset repo id (used only when `LOCAL_DATASET_DIR` is unset/empty).               |
| `TEXT_COLUMN`       | `normalized_text`          | Transcript column (HF path). Use `text` for most datasets; LJ Speech uses the normalized variant. |
| `LANGUAGE_ID`       | `en`                       | Language code stamped on every manifest row.                                                    |
| `MAX_SAMPLES`       | all (local) / `800` (HF)   | Cap on clips. Local datasets default to every clip; the LJ demo caps at 800 (~1.5h).            |
| `DEV_SIZE`          | `50`                       | Rows held out for the dev set. Set `0` to skip eval (run.sh renders a train-only data config).  |
| `MAX_WORKERS`       | `32`                       | Concurrent file downloads in `prepare.py` (HF path).                                            |
| `DATASET_SOURCE`    | `auto`                     | `auto` \| `parquet` \| `audiofolder` — how `prepare.py` materializes the wavs (HF path).        |
| `NJ_PER_GPU`        | `3`                        | Tokenizer worker processes per GPU in Stage 1.                                                  |

**Training knobs:**

The bulk of the training hyperparameters live in `config/train_config_finetune.json`. The most relevant fields (see [OmniVoice docs/training.md](https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md)):

| Field | Default | Description |
| ----- | ------- | ----------- |
| `init_from_checkpoint` | `k2-fsa/OmniVoice` | Pretrained weights to start from. |
| `steps` | `2000` | Total training steps. Sized for a small single-speaker corpus (a couple hours of audio is roughly 75 epochs at `batch_tokens=8192`); more data supports more steps, e.g. `5000` for the full LJ Speech corpus. Watch dev loss and pick the best checkpoint rather than assuming the last one wins — small corpora overfit well before the schedule ends. |
| `learning_rate` | `1e-5` | Peak LR. Lower than from-scratch training (`1e-4`). |
| `batch_tokens` | `8192` | Token budget per GPU per batch (primary memory control). |
| `eval_steps` / `save_steps` | `250` | Eval + checkpoint cadence. Kept tight so there are enough dev-loss points to select the best checkpoint on a small corpus. |
| `attn_implementation` | `flex_attention` | Attention backend. Use `config/train_config_finetune_sdpa.json` if your GPU lacks flex_attention support. |

> **SDPA and short clips.** The SDPA backend batches with length-grouped padding and drops samples outside `min_sample_tokens`..`max_sample_tokens` (flex_attention's sequence packing applies no such filter). Upstream defaults `min_sample_tokens` to 50, which would silently drop sub-second conversational utterances ("Hi.", "Bye!" — ~25-30 tokens at the codec's 25 Hz frame rate); the SDPA config here lowers it to 30 to keep them.

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

# Local utterance-level dataset:
LOCAL_DATASET_DIR=datasets/my_speaker ./run.sh

# Full LJ Speech corpus:
MAX_SAMPLES= STEPS=5000 ./run.sh

# SDPA attention (if flex_attention is unsupported on your GPU):
TRAIN_CONFIG=config/train_config_finetune_sdpa.json ./run.sh

# Multi-GPU:
GPU_IDS=0,1,2,3 NUM_GPUS=4 ./run.sh

# Different HF dataset:
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
    "exp/omnivoice_finetune/checkpoint-2000",  # pick the best dev-loss step
    device_map="cuda:0",
    dtype=torch.float16,
)

audio = model.generate(
    text="She said she would be here by noon.",
    ref_audio="hf_dataset_cache/clips/LJ001-0001.wav",  # a clip from your training corpus
    ref_text="Printing, in the only sense with which we are at present concerned, ...",
)
sf.write("output.wav", audio[0], model.sampling_rate)
```

Or via the CLI that ships with `omnivoice`:

```bash
omnivoice-infer \
  --model exp/omnivoice_finetune/checkpoint-2000 \
  --text "She said she would be here by noon." \
  --ref_audio hf_dataset_cache/clips/LJ001-0001.wav \
  --ref_text "Printing, in the only sense with which we are at present concerned, ..." \
  --output output.wav
```

### 5) Deploy a fine-tuned checkpoint with Truss

The `truss/` folder packages a trained checkpoint as a Baseten deployment. It uses the vLLM-Omni server (`docker_server` running `vllm serve --omni` on the `baseten/vllm-omni` image), which exposes an OpenAI-compatible `/v1/audio/speech` endpoint for voice cloning / voice design / auto voice.

**Files:**

- `truss/config.yaml` — Truss deployment spec (server image, GPU, checkpoint reference, start command).
- `truss/call.py` — Minimal Python client that posts text (+ optional reference clip) and writes `output.wav`.

**Point the deployment at your checkpoint**

Edit `truss/config.yaml` and set `training_job_id` to the Baseten training job that produced the checkpoint, plus the relative path you want to serve:

```22:28:examples/omnivoice-tts-finetuning/truss/config.yaml
training_checkpoints:
  download_folder: /models/training_checkpoints
  artifact_references:
    - training_job_id: qe79ppw # Replace with your training job ID
      paths:
        # OmniVoice writes checkpoint-<step> dirs; pick the step you want.
        - "rank-0/checkpoint-2000"
```

The downloaded artifacts land under `/models/training_checkpoints/<training_job_id>/<path>` inside the container. Update the `CKPT` path in the `start_command` of the same file to match (it also symlinks the base repo's `audio_tokenizer/` into the checkpoint dir, since the trainer doesn't save it):

```41:46:examples/omnivoice-tts-finetuning/truss/config.yaml
  start_command: >-
    sh -c '
    CKPT=/models/training_checkpoints/qe79ppw/rank-0/checkpoint-2000;
    ln -sfn /models/OmniVoice/audio_tokenizer "$CKPT/audio_tokenizer";
    vllm serve "$CKPT" --host 0.0.0.0 --port 8091 --trust-remote-code --omni
    '
```

Replace `qe79ppw` with your `training_job_id` and `checkpoint-2000` with whichever step you listed above.

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

OmniVoice is a zero-shot voice-clone model — fine-tuning adapts the weights toward your speaker, but the voice at inference is driven by a reference clip rather than a baked-in speaker name (unlike the Qwen3-TTS recipe). So `call.py` does **voice cloning**: it sends a short reference clip from the target speaker plus its transcript on the speech request. Set `REF_AUDIO_PATH` / `REF_TEXT` to a clip from your training corpus; when unset it falls back to pulling one LJ Speech clip (+ transcript) via the HF datasets-server, so it runs without a local copy of the training data.

Request body notes:

- `input` (required): the text to synthesize.
- `ref_audio` / `ref_text`: base64 wav + its transcript for voice cloning (what `call.py` sends). Omitting both falls back to "auto voice", which picks a *random* voice each call.
- `instruct` (optional): style attributes for voice design (e.g. `"female, british accent"`).
- `language` (optional): language name or code (e.g. `English` / `en`).
- generation knobs (optional): `num_step`, `guidance_scale`, `speed`, `duration`.
