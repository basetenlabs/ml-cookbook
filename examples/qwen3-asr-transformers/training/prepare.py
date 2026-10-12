#!/usr/bin/env python3
"""Prepare a Hugging Face audio dataset for Qwen3-ASR fine-tuning.

The upstream trainer consumes JSONL rows containing a local WAV path and a
target in Qwen3-ASR's native output format:

    {"audio": "/path/to/clip.wav",
     "text": "language English<asr_text>Hello world."}

The `language X` prefix is what the model learns to emit when it detects the
language itself, so it must name the language actually spoken in each clip.
Use `--language` for a single-language dataset, or `--language_column` to read
a per-clip label for multilingual data.

This utility downloads a dataset, resamples audio to 16 kHz mono, filters long
clips, writes deterministic train/eval manifests, and materializes the audio so
the training collator never depends on a remote URL.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


# Qwen3-ASR's canonical language names (qwen_asr.inference.utils.SUPPORTED_LANGUAGES).
# Targets must use these exact spellings: any other value is learned as a new tag
# that the serving stack does not recognise.
QWEN_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)
_CANONICAL = {name.lower(): name for name in QWEN_LANGUAGES}
# ISO 639-1/639-3 codes and common alternate names found in ASR datasets.
_ALIASES = {
    "zh": "Chinese",
    "zho": "Chinese",
    "cmn": "Chinese",
    "mandarin": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "eng": "English",
    "ar": "Arabic",
    "ara": "Arabic",
    "de": "German",
    "deu": "German",
    "fr": "French",
    "fra": "French",
    "es": "Spanish",
    "spa": "Spanish",
    "pt": "Portuguese",
    "por": "Portuguese",
    "id": "Indonesian",
    "ind": "Indonesian",
    "it": "Italian",
    "ita": "Italian",
    "ko": "Korean",
    "kor": "Korean",
    "ru": "Russian",
    "rus": "Russian",
    "th": "Thai",
    "tha": "Thai",
    "vi": "Vietnamese",
    "vie": "Vietnamese",
    "ja": "Japanese",
    "jpn": "Japanese",
    "tr": "Turkish",
    "tur": "Turkish",
    "hi": "Hindi",
    "hin": "Hindi",
    "ms": "Malay",
    "msa": "Malay",
    "nl": "Dutch",
    "nld": "Dutch",
    "sv": "Swedish",
    "swe": "Swedish",
    "da": "Danish",
    "dan": "Danish",
    "fi": "Finnish",
    "fin": "Finnish",
    "pl": "Polish",
    "pol": "Polish",
    "cs": "Czech",
    "ces": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "tagalog": "Filipino",
    "fa": "Persian",
    "fas": "Persian",
    "farsi": "Persian",
    "el": "Greek",
    "ell": "Greek",
    "ro": "Romanian",
    "ron": "Romanian",
    "hu": "Hungarian",
    "hun": "Hungarian",
    "mk": "Macedonian",
    "mkd": "Macedonian",
}
_MULTI_SEPARATORS = re.compile(r"[,+;|/]")


def canonical_language(value: Any) -> str | None:
    """Map a code or name to Qwen's spelling; None when unknown."""
    key = str(value or "").strip().lower().replace("_", "-").split("-")[0]
    return _CANONICAL.get(key) or _ALIASES.get(key)


def clip_languages(value: Any) -> list[str | None]:
    """Parse a per-clip label: one language, or several for a mixed clip.

    Accepts a string such as "es", "Spanish", or "English+Spanish", or a list.
    List mixed clips with the dominant language first.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        parts = _MULTI_SEPARATORS.split(str(value or ""))
    parts = [part for part in (p.strip() for p in parts) if part]
    return [canonical_language(part) for part in parts]


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
        help=(
            "Qwen language label for every clip of a single-language dataset. "
            "Ignored for labeled rows when --language_column is set. Qwen uses "
            "'language None' for silent audio, so an empty value is only "
            "appropriate when the data has no language labels at all."
        ),
    )
    parser.add_argument(
        "--language_column",
        default="",
        help=(
            "Column holding each clip's language (code or name, e.g. 'es' or "
            "'Spanish'; 'English+Spanish' for a mixed clip). Required for "
            "multilingual data that should keep automatic language detection."
        ),
    )
    parser.add_argument(
        "--unlabeled_policy",
        choices=("drop", "fallback", "error"),
        default="drop",
        help=(
            "With --language_column: what to do with an empty or unknown label. "
            "'fallback' uses --language, which silently mislabels non-matching "
            "clips; prefer 'drop' or 'error'."
        ),
    )
    parser.add_argument(
        "--mixed_language_policy",
        choices=("drop", "dominant"),
        default="drop",
        help=(
            "With --language_column: what to do with a clip labeled with more "
            "than one language. 'dominant' tags it with the first listed "
            "language and keeps the transcript as spoken. Qwen emits one tag "
            "per output, so a clip is never written with two tags."
        ),
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

    if args.language_column and args.language_column not in dataset.column_names:
        raise ValueError(
            f"--language_column {args.language_column!r} is not in the dataset "
            f"columns {dataset.column_names}"
        )
    fallback_language = canonical_language(args.language) if args.language else ""
    if args.language and fallback_language is None:
        raise ValueError(
            f"--language {args.language!r} is not a Qwen3-ASR language; use one "
            f"of {', '.join(QWEN_LANGUAGES)}"
        )

    rows: list[dict[str, str]] = []
    skipped_empty = 0
    skipped_duration = 0
    skipped_unlabeled = 0
    skipped_mixed = 0

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

        language = fallback_language
        if args.language_column:
            labels = clip_languages(example.get(args.language_column))
            known = [label for label in labels if label]
            if not labels or len(known) != len(labels):
                if args.unlabeled_policy == "error":
                    raise ValueError(
                        f"Row {index}: unlabeled or unsupported language "
                        f"{example.get(args.language_column)!r}"
                    )
                if args.unlabeled_policy == "drop":
                    skipped_unlabeled += 1
                    continue
            elif len(set(known)) > 1:
                if args.mixed_language_policy == "drop":
                    skipped_mixed += 1
                    continue
                language = known[0]
            else:
                language = known[0]

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
                "text": qwen_target(transcript, language),
                "language": language or "None",
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

    # Hold out evaluation rows per language, in proportion to each language's
    # share, so a minority language is not missing from the eval set.
    by_language: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_language[row["language"]].append(i)
    rng = random.Random(args.seed)
    eval_indices: set[int] = set()
    for language_rows in by_language.values():
        rng.shuffle(language_rows)
        share = round(args.eval_samples * len(language_rows) / len(rows))
        if args.eval_samples and len(language_rows) > 1:
            share = max(share, 1)
        eval_indices.update(language_rows[: min(share, len(language_rows) - 1)])
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
        f"skipped {skipped_empty} empty transcripts, "
        f"{skipped_duration} clips outside the duration bounds, "
        f"{skipped_unlabeled} unlabeled clips and {skipped_mixed} mixed-language clips."
    )
    counts = Counter(row["language"] for row in rows)
    print(
        "Clips per language: " + ", ".join(f"{k}={v}" for k, v in counts.most_common())
    )
    if len(counts) == 1 and args.language_column:
        print(
            "WARNING: every clip has the same language label; check the language "
            "column before training a multilingual model."
        )
    print(f"Train manifest: {train_path}")
    if eval_rows:
        print(f"Eval manifest:  {eval_path}")


if __name__ == "__main__":
    main()
