## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning. This recipe uses [LJ Speech](https://huggingface.co/datasets/SeanSleat/lj_speech) (~24 hours of single-speaker English from a public-domain reader, 13.1k clips) as the running example, but any HF dataset that exposes a text column and an audio column will work.

**Note:** The `run.sh` script automatically creates and uses a virtual environment (`.venv`) to avoid conflicts with system Python installations. If running scripts manually, ensure you're using a virtual environment or have the required packages installed.

### 0) Prepare the dataset

`prepare.py` downloads a HuggingFace TTS dataset, materializes the wav clips locally, and writes the training JSONL with precomputed `audio_codes` so the actual SFT loop stays fast.

For LJ Speech (`SeanSleat/lj_speech`), the relevant columns are:

| Column            | Type   | Description                                                         |
|-------------------|--------|---------------------------------------------------------------------|
| `id`              | string | Clip id, e.g. `LJ001-0001`. Used to name the extracted wav files.   |
| `audio`           | audio  | Inline audio bytes (auto-extracted into `{cache_dir}/clips/*.wav`). |
| `text`            | string | Raw transcript ("printed in 1462").                                 |
| `normalized_text` | string | Numbers/dates spelled out ("printed in fourteen sixty-two").        |

For LJ Speech we pass `--text_column normalized_text` because the audio matches the normalized reading.

In general the script needs:
- a text column (any name, picked via `--text_column`, default `text`)
- one of: an `audio` column with embedded bytes (parquet path), or a `file_name` column pointing at a wav in the repo (AudioFolder path)
- optionally an `id` column, used to name the extracted wavs when there is no `file_name`

Because the model only supports single-speaker fine-tuning, the first sample (after sorting) is reused as the shared `ref_audio` for every row, which the Qwen3-TTS docs recommend for speaker consistency. Use `--ref_index` to pick a different one.

If your dataset is gated, make sure you're authenticated by setting your `hf_access_token` secret in the Baseten UI.

**Download + tokenize (LJ Speech):**
```bash
python prepare.py \
  --dataset_repo SeanSleat/lj_speech \
  --text_column normalized_text \
  --output_jsonl train.jsonl \
  --cache_dir ./hf_dataset_cache
```

**Common options:**
- `--dataset_repo` (required): HuggingFace dataset repo id.
- `--text_column`: Transcript column to use (default: `text`; use `normalized_text` for LJ Speech).
- `--output_jsonl`: Final training JSONL with `audio_codes` (default: `train.jsonl`).
- `--cache_dir`: Local directory for the dataset snapshot (default: `./hf_dataset_cache`).
- `--max_samples`: Cap the number of training rows (default: all). `run.sh` defaults this to `800` (~1.5h of LJ Speech audio).
- `--ref_index`: Which sample (after sorting) to reuse as `ref_audio` for every row (default: `0`).
- `--hf_token`: Override the Hugging Face token used for the download.
- `--max_workers`: Number of concurrent file downloads (default: `32`).
- `--source`: `auto` (default), `parquet`, or `audiofolder`. See "Download speed" below.
- `--device`: Device for the Qwen3 TTS tokenizer pass (default: `cuda:0`).
- `--tokenizer_model_path`: Tokenizer repo/path (default: `Qwen/Qwen3-TTS-Tokenizer-12Hz`).
- `--skip_tokenize`: Skip the `audio_codes` step (handy for inspecting the manifest without loading the tokenizer model).

**One-click script:**
```bash

# Defaults: LJ Speech + normalized_text, capped at 800 clips (~1.5h of audio).
./run.sh

# Smaller subset for a quick smoke test:
MAX_SAMPLES=200 ./run.sh

# Use the full dataset (no cap):
MAX_SAMPLES= ./run.sh

# Use a different dataset / text column:
DATASET_REPO=org/my-tts-dataset TEXT_COLUMN=text ./run.sh
```

### 1) Training JSONL format

After `prepare.py` runs, `train.jsonl` contains one JSON object per line:

- `audio`: path to the target training audio (wav) inside `./hf_dataset_cache/clips/...`
- `text`: transcript corresponding to `audio` (from `--text_column`)
- `ref_audio`: path to the reference speaker audio (wav) — same value on every line
- `audio_codes`: precomputed discrete codec codes for `audio` (used directly by the SFT loop)

Example (with `audio_codes` truncated for readability):
```jsonl
{"audio":"/abs/path/clips/LJ001-0001.wav","text":"Printing, in the only sense with which we are at present concerned, ...","ref_audio":"/abs/path/clips/LJ001-0001.wav","audio_codes":[[...]]}
{"audio":"/abs/path/clips/LJ001-0002.wav","text":"in being comparatively modern.","ref_audio":"/abs/path/clips/LJ001-0001.wav","audio_codes":[[...]]}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples. Ensure that it is a neutral and well phrased reference.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.

### 2) Fine-tune

Run the end-to-end pipeline (dataset prep + SFT):

```bash
./run.sh
```

`run.sh` is preset for a single H100 on ~1.5h of audio (~800 LJ Speech clips). The defaults below are a good starting point; override any of them via env vars (e.g. `LR=1e-5 EPOCHS=8 ./run.sh`) or by editing the script.

**Key training hyperparameters:**

| Variable               | Default | Description                                                                                                                                                |
|------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `BATCH_SIZE`           | `4`     | Per-device micro-batch size. Lower this first if you OOM.                                                                                                  |
| `GRAD_ACCUM`           | `2`     | Gradient accumulation steps. Effective batch size = `BATCH_SIZE * GRAD_ACCUM` (= 8 by default).                                                            |
| `LR`                   | `5e-6`  | Peak learning rate for AdamW. TTS fine-tuning is sensitive — values much above `1e-5` tend to degrade speaker quality.                                     |
| `EPOCHS`               | `12`    | Number of passes over the training set. For ~800 clips, 8–16 epochs is typical; watch eval loss to pick a checkpoint.                                      |
| `WARMUP_RATIO`         | `0.05`  | Fraction of total optimizer steps used to linearly warm up the LR before the cosine decay kicks in.                                                        |
| `SAVE_EVERY_N_EPOCHS`  | `2`     | Checkpoint cadence. Checkpoints land in `${BT_CHECKPOINT_DIR:-output}/checkpoint-epoch-N`.                                                                 |
| `EVAL_SPLIT`           | `40`    | Number of rows held out (deterministically) for per-epoch validation loss. Set to `0` to disable eval.                                                     |
| `SPEAKER_NAME`         | `ft_speaker` | Label baked into the saved model's speaker registry; pass this same string to `generate_custom_voice(speaker=...)` at inference time.                 |

**Data / dataset knobs:**

| Variable          | Default                  | Description                                                                                          |
|-------------------|--------------------------|------------------------------------------------------------------------------------------------------|
| `DATASET_REPO`    | `SeanSleat/lj_speech`    | HuggingFace dataset repo id passed to `prepare.py`.                                                  |
| `TEXT_COLUMN`     | `normalized_text`        | Transcript column. Use `text` for most datasets; LJ Speech uses the normalized variant.              |
| `MAX_SAMPLES`     | `800`                    | Cap on training rows (~1.5h of LJ Speech). Set `MAX_SAMPLES=` (empty) to use the full dataset.       |
| `MAX_WORKERS`     | `32`                     | Concurrent file downloads in `prepare.py`.                                                           |
| `DATASET_SOURCE`  | `auto`                   | `auto` \| `parquet` \| `audiofolder` — how `prepare.py` materializes the wavs.                       |

**Model / output paths:**

| Variable             | Default                              | Description                                                                                                                          |
|----------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `INIT_MODEL_PATH`    | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`      | Base model. On Baseten this is set to the pre-mounted weights dir; locally it falls back to the HF repo id and snapshot-downloads.   |
| `OUTPUT_DIR`         | `${BT_CHECKPOINT_DIR:-output}`       | Where checkpoints are written. On Baseten, only `$BT_CHECKPOINT_DIR` is persisted across pod tear-down.                              |
| `TOKENIZER_MODEL_PATH` | `Qwen/Qwen3-TTS-Tokenizer-12Hz`    | Audio codec tokenizer used by `prepare.py` to precompute `audio_codes`.                                                              |

**Common overrides:**

```bash
# Quick smoke test (smaller subset, fewer epochs):
MAX_SAMPLES=200 EPOCHS=4 ./run.sh

# Train on the full LJ Speech corpus:
MAX_SAMPLES= ./run.sh

# Different dataset:
DATASET_REPO=org/my-tts-dataset TEXT_COLUMN=text ./run.sh

# Lower memory footprint:
BATCH_SIZE=2 GRAD_ACCUM=4 ./run.sh
```

### 3) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="ljspeech",
)
sf.write("output.wav", wavs[0], sr)
```