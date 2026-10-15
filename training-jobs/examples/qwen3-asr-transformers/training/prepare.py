#!/usr/bin/env python3
"""Prepare a Hugging Face audio dataset for Qwen3-ASR fine-tuning.

The upstream trainer consumes JSONL rows containing a local WAV path and a
target in Qwen3-ASR's native output format:

    {"audio": "/path/to/clip.wav",
     "text": "language English<asr_text>Hello world."}

This utility downloads a dataset, resamples audio to 16 kHz mono, filters long
clips, writes deterministic train/eval manifests, and materializes the audio so
the training collator never depends on a remote URL.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


def safe_filename(value: Any, index: int) -> str:
    candidate = str(value) if value not in (None, "") else f"sample-{index:07d}"
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-._")
    return candidate or f"sample-{index:07d}"


def qwen_target(transcript: str, language: str) -> str:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty")
    return f"language {language or 'None'}<asr_text>{transcript}"


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a Hugging Face ASR dataset as WAV + Qwen JSONL."
    )
    parser.add_argument(
        "--dataset_repo",
        default="openslr/librispeech_asr",
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--dataset_config",
        default="clean",
        help="Optional dataset configuration/subset. Pass an empty string for none.",
    )
    parser.add_argument("--dataset_split", default="train.100")
    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--id_column", default="id")
    parser.add_argument(
        "--language",
        default="English",
        help="Qwen language label. Use an empty value to write 'language None'.",
    )
    parser.add_argument("--sampling_rate", type=int, default=16_000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_duration_seconds", type=float, default=0.1)
    parser.add_argument("--max_duration_seconds", type=float, default=30.0)
    parser.add_argument("--eval_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="./cache/qwen3-asr-dataset")
    parser.add_argument("--train_jsonl", default="train.jsonl")
    parser.add_argument("--eval_jsonl", default="eval.jsonl")
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token; defaults to HF_TOKEN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    clips_dir = cache_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset_repo,
        args.dataset_config or None,
        split=args.dataset_split,
        cache_dir=str(cache_dir / "datasets"),
        token=args.hf_token,
    )

    missing = {
        column
        for column in (args.audio_column, args.text_column)
        if column not in dataset.column_names
    }
    if missing:
        raise ValueError(
            f"Dataset columns are {dataset.column_names}; missing required "
            f"column(s): {sorted(missing)}"
        )

    dataset = dataset.cast_column(
        args.audio_column,
        Audio(sampling_rate=args.sampling_rate, mono=True),
    )

    rows: list[dict[str, str]] = []
    skipped_empty = 0
    skipped_duration = 0

    for index, example in enumerate(dataset):
        transcript = str(example.get(args.text_column) or "").strip()
        if not transcript:
            skipped_empty += 1
            continue

        audio = example[args.audio_column]
        samples = np.asarray(audio["array"], dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=0)
        sample_rate = int(audio["sampling_rate"])
        duration = len(samples) / sample_rate
        if not args.min_duration_seconds <= duration <= args.max_duration_seconds:
            skipped_duration += 1
            continue

        example_id = (
            example.get(args.id_column)
            if args.id_column and args.id_column in dataset.column_names
            else None
        )
        stem = safe_filename(example_id, index)
        wav_path = clips_dir / f"{index:07d}-{stem}.wav"
        sf.write(wav_path, samples, sample_rate, subtype="PCM_16")

        rows.append(
            {
                "audio": str(wav_path),
                "text": qwen_target(transcript, args.language),
            }
        )
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    if not rows:
        raise RuntimeError("No usable examples remained after dataset filtering.")
    if args.eval_samples < 0:
        raise ValueError("--eval_samples must be non-negative.")
    if args.eval_samples >= len(rows):
        raise ValueError(
            f"--eval_samples={args.eval_samples} must be smaller than the "
            f"{len(rows)} prepared examples."
        )

    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    eval_indices = set(indices[: args.eval_samples])
    train_rows = [row for i, row in enumerate(rows) if i not in eval_indices]
    eval_rows = [row for i, row in enumerate(rows) if i in eval_indices]

    train_path = Path(args.train_jsonl).expanduser().resolve()
    eval_path = Path(args.eval_jsonl).expanduser().resolve()
    write_jsonl(train_path, train_rows)
    if eval_rows:
        write_jsonl(eval_path, eval_rows)
    elif eval_path.exists():
        eval_path.unlink()

    print(
        "Prepared "
        f"{len(train_rows)} train / {len(eval_rows)} eval examples; "
        f"skipped {skipped_empty} empty transcripts and "
        f"{skipped_duration} clips outside the duration bounds."
    )
    print(f"Train manifest: {train_path}")
    if eval_rows:
        print(f"Eval manifest:  {eval_path}")


if __name__ == "__main__":
    main()
