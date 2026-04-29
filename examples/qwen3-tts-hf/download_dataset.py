#!/usr/bin/env python3
"""
Download the baseten-admin/sierra-ft-tts dataset from Hugging Face and
convert it into the JSONL format required by Qwen3-TTS finetuning.

Output JSONL fields (one JSON object per line):
- audio:     path to the target training audio (wav)
- text:      transcript corresponding to audio
- ref_audio: path to the reference speaker audio (wav, same for every line)

Source dataset schema (https://huggingface.co/datasets/baseten-admin/sierra-ft-tts):
- file_name      str    Relative path to the audio clip (e.g. clips/0001.wav)
- text           str    Punctuated transcript for the clip
- start / end    float  Source-audio offsets in seconds
- duration       float  Clip duration in seconds
- num_words      int    Number of Deepgram words in the sentence
- avg_confidence float  Mean Deepgram word confidence

Speed strategy:
The script first tries to download HuggingFace's auto-converted parquet
shards (a handful of large files, much faster with `hf_transfer`). The audio
bytes are embedded inline in the parquet rows, so we extract them locally to
wav files and skip the per-clip HTTP roundtrips entirely. If parquet shards
aren't available, it falls back to a two-pass AudioFolder download:
metadata-only first, then exactly the wav clips that survive filtering.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

DATASET_REPO = "baseten-admin/sierra-ft-tts"

# `allow_patterns` for the metadata-only pre-pass on the AudioFolder fallback
# path. We grab anything that could plausibly be a manifest plus README/json
# config files (they're tiny). Wav clips are pulled in a separate, filtered
# second pass.
METADATA_ALLOW_PATTERNS = [
    "metadata.csv",
    "metadata.jsonl",
    "*.parquet",
    "*.json",
    "*.md",
    "README*",
]


# ---------- helpers shared by both download paths ----------


def _coerce_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_passes_filters(
    row: dict,
    min_confidence: float,
    min_duration: float,
    max_duration: float,
) -> bool:
    confidence = _coerce_float(row.get("avg_confidence"), default=1.0)
    duration = _coerce_float(row.get("duration"), default=None)
    if confidence is not None and confidence < min_confidence:
        return False
    if duration is not None:
        if duration < min_duration:
            return False
        if max_duration > 0 and duration > max_duration:
            return False
    return True


def _ensure_hf_transfer_or_warn() -> None:
    import importlib.util

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").lower() not in ("1", "true", "yes"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if importlib.util.find_spec("hf_transfer") is None:
        print(
            "[warn] HF_HUB_ENABLE_HF_TRANSFER=1 is set but the `hf_transfer` "
            "package is not importable. Falling back to the slow pure-Python "
            "downloader. Install with `pip install hf-transfer` to fix."
        )
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)


# ---------- fast path: parquet shards from refs/convert/parquet ----------


def _try_download_parquet_shards(
    cache_dir: Path,
    token: Optional[str],
    max_workers: int,
) -> Optional[Path]:
    """
    Try to download HF's auto-converted parquet shards. Returns the local repo
    root if successful, else None. Auto-convert lives on the special
    `refs/convert/parquet` git ref and contains shards like
    `default/train/0000.parquet`.
    """
    parquet_root = cache_dir / "parquet"
    parquet_root.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision="refs/convert/parquet",
            local_dir=str(parquet_root),
            token=token,
            allow_patterns=["**/*.parquet"],
            max_workers=max_workers,
        )
    except (RevisionNotFoundError, EntryNotFoundError):
        return None
    except RepositoryNotFoundError:
        return None
    except Exception as e:
        print(f"[warn] parquet shard download failed: {e}")
        return None

    if not list(Path(path).rglob("*.parquet")):
        return None
    return Path(path)


def _extract_from_parquet(
    parquet_root: Path,
    clips_dir: Path,
    min_confidence: float,
    min_duration: float,
    max_duration: float,
) -> List[dict]:
    """
    Stream rows out of every parquet shard, apply filters, and write the
    embedded audio bytes to local wav files.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required to read parquet shards. "
            "Install with: pip install pyarrow"
        ) from e

    clips_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(parquet_root.rglob("*.parquet"))
    print(f"Reading {len(parquet_files)} parquet shard(s)...")

    rows: List[dict] = []
    written = 0
    started = time.time()
    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        col_names = table.column_names

        if "audio" not in col_names:
            raise RuntimeError(
                f"Parquet shard {pq_file.name} has no 'audio' column "
                f"(found: {col_names})"
            )
        if "text" not in col_names:
            raise RuntimeError(
                f"Parquet shard {pq_file.name} has no 'text' column "
                f"(found: {col_names})"
            )

        for record in table.to_pylist():
            if not _row_passes_filters(
                record,
                min_confidence=min_confidence,
                min_duration=min_duration,
                max_duration=max_duration,
            ):
                continue

            audio = record["audio"]
            if isinstance(audio, dict):
                audio_bytes = audio.get("bytes")
                audio_path_in_repo = audio.get("path")
            else:
                audio_bytes = audio
                audio_path_in_repo = None

            if not audio_bytes:
                continue

            file_name = (
                record.get("file_name")
                or audio_path_in_repo
                or f"clips/{written:06d}.wav"
            )
            out_path = (clips_dir / file_name).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                written += 1

            rows.append({
                "file_name": file_name,
                "audio_path": str(out_path),
                "text": record["text"],
                "duration": _coerce_float(record.get("duration"), default=None),
                "avg_confidence": _coerce_float(
                    record.get("avg_confidence"), default=1.0
                ),
            })

    elapsed = max(time.time() - started, 1e-3)
    print(
        f"Extracted {len(rows)} rows ({written} new wav files written) "
        f"in {elapsed:.1f}s ({written / elapsed:.0f} files/sec written)."
    )
    return rows


# ---------- fallback path: AudioFolder (metadata + per-clip wav download) ----------


def _iter_audiofolder_metadata(repo_root: Path) -> Iterator[dict]:
    """
    Yield metadata rows from the dataset snapshot for the AudioFolder layout.
    Tries metadata.jsonl -> metadata.csv -> any *.parquet at the root.
    """
    jsonl_path = repo_root / "metadata.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    csv_path = repo_root / "metadata.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield row
        return

    parquet_files = sorted(repo_root.glob("*.parquet"))
    if parquet_files:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "Found parquet metadata but pyarrow is not installed. "
                "Install with: pip install pyarrow"
            ) from e
        for pf in parquet_files:
            table = pq.read_table(pf)
            for row in table.to_pylist():
                yield row
        return

    raise FileNotFoundError(
        f"No metadata file (metadata.jsonl, metadata.csv, or *.parquet) "
        f"found in {repo_root}"
    )


def _filter_audiofolder_rows(
    rows: Iterable[dict],
    repo_root: Path,
    min_confidence: float,
    min_duration: float,
    max_duration: float,
) -> Iterator[dict]:
    for row in rows:
        file_name = row.get("file_name")
        text = row.get("text")
        if not file_name or not text:
            continue
        if not _row_passes_filters(
            row,
            min_confidence=min_confidence,
            min_duration=min_duration,
            max_duration=max_duration,
        ):
            continue
        yield {
            "file_name": file_name,
            "audio_path": str((repo_root / file_name).resolve()),
            "text": text,
            "duration": _coerce_float(row.get("duration"), default=None),
            "avg_confidence": _coerce_float(
                row.get("avg_confidence"), default=1.0
            ),
        }


def _download_clips(
    cache_dir: Path,
    file_names: List[str],
    token: Optional[str],
    max_workers: int,
) -> None:
    if not file_names:
        return
    started = time.time()
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(cache_dir),
        token=token,
        allow_patterns=file_names,
        max_workers=max_workers,
    )
    elapsed = max(time.time() - started, 1e-3)
    print(
        f"Pulled {len(file_names)} clip(s) in {elapsed:.1f}s "
        f"({len(file_names) / elapsed:.0f} files/sec, max_workers={max_workers})."
    )


def _download_via_audiofolder(
    cache_dir: Path,
    token: Optional[str],
    max_workers: int,
    min_confidence: float,
    min_duration: float,
    max_duration: float,
) -> Tuple[Path, List[dict]]:
    print(f"[fallback] Fetching metadata from {DATASET_REPO} (AudioFolder layout)...")
    repo_root = Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=token,
            allow_patterns=METADATA_ALLOW_PATTERNS,
            max_workers=max_workers,
        )
    )

    rows = list(
        _filter_audiofolder_rows(
            _iter_audiofolder_metadata(repo_root),
            repo_root=repo_root,
            min_confidence=min_confidence,
            min_duration=min_duration,
            max_duration=max_duration,
        )
    )
    return repo_root, rows


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download baseten-admin/sierra-ft-tts and convert it to a JSONL "
            "for Qwen3-TTS finetuning."
        )
    )
    parser.add_argument("--output_jsonl", type=str, default="train_raw.jsonl")
    parser.add_argument("--cache_dir", type=str, default="./hf_dataset_cache")
    parser.add_argument(
        "--ref_index",
        type=int,
        default=0,
        help="Index of the sample to use as shared reference audio (default: 0).",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--min_duration", type=float, default=0.0)
    parser.add_argument(
        "--max_duration",
        type=float,
        default=0.0,
        help="0 disables the upper bound (default: 0).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help=(
            "Concurrent file downloads passed to snapshot_download. "
            "32 is a good default; 64+ can help on fast networks (default: 32)."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "parquet", "audiofolder"],
        help=(
            "Where to pull data from. `parquet` uses HF's auto-converted "
            "parquet shards (much faster, embeds audio inline). `audiofolder` "
            "downloads metadata + each wav as a separate file. `auto` tries "
            "parquet first and falls back to audiofolder (default: auto)."
        ),
    )
    args = parser.parse_args()

    _ensure_hf_transfer_or_warn()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_root: Optional[Path] = None
    rows: Optional[List[dict]] = None

    if args.source in ("auto", "parquet"):
        print(f"Trying parquet shard fast-path for {DATASET_REPO}...")
        parquet_root = _try_download_parquet_shards(
            cache_dir=cache_dir,
            token=args.hf_token,
            max_workers=args.max_workers,
        )
        if parquet_root is not None:
            rows = _extract_from_parquet(
                parquet_root=parquet_root,
                clips_dir=cache_dir,
                min_confidence=args.min_confidence,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
            repo_root = cache_dir
        else:
            if args.source == "parquet":
                raise SystemExit(
                    "Parquet shards are not available for this dataset "
                    "(refs/convert/parquet is empty or inaccessible). "
                    "Re-run with --source audiofolder."
                )
            print("[info] No parquet shards available, falling back to AudioFolder download.")

    if rows is None:
        repo_root, rows = _download_via_audiofolder(
            cache_dir=cache_dir,
            token=args.hf_token,
            max_workers=args.max_workers,
            min_confidence=args.min_confidence,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )

    if not rows:
        raise SystemExit(
            "No usable rows after filtering. Check --min_confidence / "
            "--min_duration / --max_duration thresholds."
        )

    rows.sort(key=lambda r: r["file_name"])
    total_after_filter = len(rows)

    if args.max_samples is not None and args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not 0 <= args.ref_index < len(rows):
        raise SystemExit(
            f"--ref_index {args.ref_index} is out of range "
            f"(only {len(rows)} samples available)."
        )

    print(
        f"Filtered samples: {total_after_filter}"
        + (f" (using first {len(rows)})" if len(rows) != total_after_filter else "")
    )

    # AudioFolder path needs a follow-up wav download for the surviving rows.
    # Parquet path already wrote the wavs while extracting, so this is a no-op
    # in that case.
    missing = [r for r in rows if not Path(r["audio_path"]).exists()]
    if missing:
        needed_files = sorted({r["file_name"] for r in missing})
        print(
            f"Downloading {len(needed_files)} audio clips with "
            f"max_workers={args.max_workers}..."
        )
        _download_clips(
            cache_dir=cache_dir,
            file_names=needed_files,
            token=args.hf_token,
            max_workers=args.max_workers,
        )

    still_missing = [r for r in rows if not Path(r["audio_path"]).exists()]
    if still_missing:
        raise SystemExit(
            f"Expected audio file(s) missing after download, e.g. "
            f"{still_missing[0]['file_name']!r}. Check --hf_token and dataset access."
        )

    ref_audio_path = rows[args.ref_index]["audio_path"]
    print(f"Reference audio: {ref_audio_path}")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            entry = {
                "audio": row["audio_path"],
                "text": row["text"],
                "ref_audio": ref_audio_path,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✓ Wrote {len(rows)} rows to {output_path}")
    print(f"  - Dataset cache: {repo_root}")
    print(f"  - Reference audio (shared by every row): {ref_audio_path}")


if __name__ == "__main__":
    main()
