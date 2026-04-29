## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning.

**Note:** The `run.sh` script automatically creates and uses a virtual environment (`.venv`) to avoid conflicts with system Python installations. If running scripts manually, ensure you're using a virtual environment or have the required packages installed.

### 0) Download Dataset from Hugging Face

This example downloads and prepares data from the [baseten-admin/sierra-ft-tts](https://huggingface.co/datasets/baseten-admin/sierra-ft-tts) dataset.

`sierra-ft-tts` is a single-speaker dataset with the following schema:

| Column           | Type   | Description                                                |
|------------------|--------|------------------------------------------------------------|
| `file_name`      | string | Relative path to the audio clip (e.g. `clips/0001.wav`).   |
| `text`           | string | Punctuated transcript for the clip.                        |
| `start` / `end`  | float  | Source-audio offsets in seconds.                           |
| `duration`       | float  | Clip duration in seconds.                                  |
| `num_words`      | int    | Number of Deepgram words in the sentence.                  |
| `avg_confidence` | float  | Mean Deepgram word confidence.                             |

Because it is single-speaker, there is no voice filter — the script just downloads everything and writes a JSONL pointing at the local wav clips. The first sample (after sorting + filtering) is reused as the shared `ref_audio` for every row, which the Qwen3-TTS docs recommend for speaker consistency.

The dataset is gated, so make sure you are authenticated. Either:

- run `hf auth login` once and let the script use your cached token, or
- export `HF_TOKEN=hf_xxx` before running the script, or
- pass `--hf_token hf_xxx` directly.

**Download and convert:**
```bash
python download_dataset.py \
  --output_jsonl train_raw.jsonl \
  --cache_dir ./hf_dataset_cache
```

**Common options:**
- `--output_jsonl`: Output JSONL file path (default: `train_raw.jsonl`).
- `--cache_dir`: Local directory for the dataset snapshot (default: `./hf_dataset_cache`).
- `--max_samples`: Cap the number of training rows (default: all).
- `--min_confidence`: Drop rows whose `avg_confidence` is below this value (default: `0.0`).
- `--min_duration` / `--max_duration`: Filter clips by `duration` in seconds. `--max_duration 0` (the default) disables the upper bound.
- `--ref_index`: Which sample (after sorting + filtering) to reuse as `ref_audio` for every row (default: `0`).
- `--hf_token`: Override the Hugging Face token used for the download.
- `--max_workers`: Number of concurrent file downloads (default: `32`).
- `--source`: `auto` (default), `parquet`, or `audiofolder`. See "Download speed" below.

**Download speed:** the script defaults to `--source auto`, which tries the *parquet fast-path* first and only falls back to the per-clip wav download if that's not available.

- **Parquet fast-path (`--source parquet`):** pulls HuggingFace's auto-converted parquet shards from the `refs/convert/parquet` branch. These are a handful of large files (with the audio bytes embedded inline), which means **few HTTP requests + multipart parallelism via `hf_transfer`** — typically 5-10x faster than fetching every wav individually. The script extracts the audio bytes locally to `{cache_dir}/clips/*.wav`. Requires `pyarrow` (already in `requirements.txt`).
- **AudioFolder fallback (`--source audiofolder`):** two-pass download — first the metadata only, then exactly the wav clips that survive your `--min_confidence` / `--min_duration` / `--max_duration` / `--max_samples` filters, in parallel.

`HF_HUB_ENABLE_HF_TRANSFER=1` is set automatically. If `hf_transfer` isn't importable, the script prints a warning and falls back to pure-Python downloads (much slower) — install with `pip install hf-transfer`.

**One-click script:**
```bash
# Run the whole pipeline (download -> prepare_data -> sft_12hz)
./run.sh

# With optional quality / size / speed knobs:
MAX_SAMPLES=500 MIN_CONFIDENCE=0.85 MAX_DURATION=15 MAX_WORKERS=64 ./run.sh

# Force a particular download path (useful for debugging):
DATASET_SOURCE=parquet ./run.sh         # fast-path only, fail if unavailable
DATASET_SOURCE=audiofolder ./run.sh     # skip parquet, do per-clip downloads
```

### 1) Input JSONL format

After `download_dataset.py` runs, `train_raw.jsonl` contains one JSON object per line:

- `audio`: path to the target training audio (wav) inside `./hf_dataset_cache/clips/...`
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav) — same value on every line

Example:
```jsonl
{"audio":"/abs/path/hf_dataset_cache/clips/0001.wav","text":"She said she would be here by noon.","ref_audio":"/abs/path/hf_dataset_cache/clips/0001.wav"}
{"audio":"/abs/path/hf_dataset_cache/clips/0002.wav","text":"Alright, let me check that for you.","ref_audio":"/abs/path/hf_dataset_cache/clips/0001.wav"}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```


### 3) Fine-tune

Run SFT using the prepared JSONL:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name sierra
```

Checkpoints will be written to:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...


### 4) Quick inference test

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
    speaker="sierra",
)
sf.write("output.wav", wavs[0], sr)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="sierra"

python download_dataset.py \
  --output_jsonl ${RAW_JSONL} \
  --cache_dir ./hf_dataset_cache

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```
