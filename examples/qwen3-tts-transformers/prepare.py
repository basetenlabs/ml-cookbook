#!/usr/bin/env python3
"""
Download a single-speaker TTS dataset from Hugging Face, tokenize the audio
with the Qwen3-TTS tokenizer, and write a training JSONL.

Pipeline:
  HF dataset  ->  local wav files  ->  Qwen3TTSTokenizer.encode  ->  output JSONL

Output JSONL fields (one JSON object per line):
- audio:       absolute path to the target training audio (wav)
- text:        transcript corresponding to audio
- ref_audio:   absolute path to the reference speaker audio (wav, same for every line)
- audio_codes: precomputed discrete codec codes for `audio`

Expected source dataset schema:
- A text column (default `text`, override with --text_column; useful for
  datasets that provide e.g. `normalized_text` alongside `text`).
- Either an `audio` column with embedded bytes (parquet path), or a
  `file_name` column pointing at a wav file in the repo (AudioFolder path).
- If neither `file_name` nor a usable `audio.path` is available, an `id`
  column is used to name the extracted wav (e.g. `clips/<id>.wav`).

Download speed strategy:
The script first tries to download HuggingFace's auto-converted parquet
shards (a handful of large files, much faster with `hf_transfer`). The audio
bytes are embedded inline in the parquet rows, so we extract them locally to
wav files and skip the per-clip HTTP roundtrips entirely. If parquet shards
aren't available, it falls back to a two-pass AudioFolder download:
metadata-only first, then exactly the wav clips that survived sampling.
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

BATCH_INFER_NUM = 32

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
    dataset_repo: str,
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
            repo_id=dataset_repo,
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
    text_column: str,
) -> List[dict]:
    """
    Stream rows out of every parquet shard and write the embedded audio bytes
    to local wav files.
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
        if text_column not in col_names:
            raise RuntimeError(
                f"Parquet shard {pq_file.name} has no '{text_column}' column "
                f"(found: {col_names})"
            )

        for record in table.to_pylist():
            text = record.get(text_column)
            if not text:
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

            row_id = record.get("id")
            if record.get("file_name"):
                file_name = record["file_name"]
            elif row_id:
                file_name = f"clips/{row_id}.wav"
            elif audio_path_in_repo:
                file_name = audio_path_in_repo
            else:
                file_name = f"clips/{written:06d}.wav"

            out_path = (clips_dir / file_name).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                written += 1

            rows.append({
                "file_name": file_name,
                "audio_path": str(out_path),
                "text": text,
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


def _normalize_audiofolder_rows(
    rows: Iterable[dict],
    repo_root: Path,
    text_column: str,
) -> Iterator[dict]:
    for row in rows:
        file_name = row.get("file_name")
        text = row.get(text_column)
        if not file_name or not text:
            continue
        yield {
            "file_name": file_name,
            "audio_path": str((repo_root / file_name).resolve()),
            "text": text,
        }


def _download_clips(
    dataset_repo: str,
    cache_dir: Path,
    file_names: List[str],
    token: Optional[str],
    max_workers: int,
) -> None:
    if not file_names:
        return
    started = time.time()
    snapshot_download(
        repo_id=dataset_repo,
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
    dataset_repo: str,
    cache_dir: Path,
    token: Optional[str],
    max_workers: int,
    text_column: str,
) -> Tuple[Path, List[dict]]:
    print(f"[fallback] Fetching metadata from {dataset_repo} (AudioFolder layout)...")
    repo_root = Path(
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=token,
            allow_patterns=METADATA_ALLOW_PATTERNS,
            max_workers=max_workers,
        )
    )

    rows = list(
        _normalize_audiofolder_rows(
            _iter_audiofolder_metadata(repo_root),
            repo_root=repo_root,
            text_column=text_column,
        )
    )
    return repo_root, rows


# ---------- download orchestration ----------


def download_dataset(
    dataset_repo: str,
    cache_dir: Path,
    token: Optional[str],
    max_workers: int,
    source: str,
    max_samples: Optional[int],
    ref_index: int,
    text_column: str,
) -> Tuple[List[dict], str]:
    """
    Materialize wav clips on disk and return:
      - rows: list of {audio, text, ref_audio} dicts (paths absolute)
      - ref_audio_path: the absolute path used as the shared reference audio
    """
    repo_root: Optional[Path] = None
    rows: Optional[List[dict]] = None

    if source in ("auto", "parquet"):
        print(f"Trying parquet shard fast-path for {dataset_repo}...")
        parquet_root = _try_download_parquet_shards(
            dataset_repo=dataset_repo,
            cache_dir=cache_dir,
            token=token,
            max_workers=max_workers,
        )
        if parquet_root is not None:
            rows = _extract_from_parquet(
                parquet_root=parquet_root,
                clips_dir=cache_dir,
                text_column=text_column,
            )
            repo_root = cache_dir
        else:
            if source == "parquet":
                raise SystemExit(
                    "Parquet shards are not available for this dataset "
                    "(refs/convert/parquet is empty or inaccessible). "
                    "Re-run with --source audiofolder."
                )
            print("[info] No parquet shards available, falling back to AudioFolder download.")

    if rows is None:
        repo_root, rows = _download_via_audiofolder(
            dataset_repo=dataset_repo,
            cache_dir=cache_dir,
            token=token,
            max_workers=max_workers,
            text_column=text_column,
        )

    if not rows:
        raise SystemExit("No usable rows found in the dataset.")

    rows.sort(key=lambda r: r["file_name"])
    total_rows = len(rows)

    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    if not 0 <= ref_index < len(rows):
        raise SystemExit(
            f"--ref_index {ref_index} is out of range "
            f"(only {len(rows)} samples available)."
        )

    print(
        f"Samples: {total_rows}"
        + (f" (using first {len(rows)})" if len(rows) != total_rows else "")
    )

    # AudioFolder path needs a follow-up wav download for the surviving rows.
    # Parquet path already wrote the wavs while extracting, so this is a no-op
    # in that case.
    missing = [r for r in rows if not Path(r["audio_path"]).exists()]
    if missing:
        needed_files = sorted({r["file_name"] for r in missing})
        print(
            f"Downloading {len(needed_files)} audio clips with "
            f"max_workers={max_workers}..."
        )
        _download_clips(
            dataset_repo=dataset_repo,
            cache_dir=cache_dir,
            file_names=needed_files,
            token=token,
            max_workers=max_workers,
        )

    still_missing = [r for r in rows if not Path(r["audio_path"]).exists()]
    if still_missing:
        raise SystemExit(
            f"Expected audio file(s) missing after download, e.g. "
            f"{still_missing[0]['file_name']!r}. Check --hf_token and dataset access."
        )

    ref_audio_path = rows[ref_index]["audio_path"]
    print(f"Reference audio: {ref_audio_path}")
    print(f"Dataset cache: {repo_root}")

    return (
        [
            {
                "audio": row["audio_path"],
                "text": row["text"],
                "ref_audio": ref_audio_path,
            }
            for row in rows
        ],
        ref_audio_path,
    )


# ---------- tokenize: precompute audio_codes ----------


def tokenize_audio_codes(
    rows: List[dict],
    tokenizer_model_path: str,
    device: str,
) -> List[dict]:
    """
    Run the Qwen3 TTS tokenizer over every row's `audio` field in batches and
    attach the resulting discrete codes as `audio_codes`.
    """
    from qwen_tts import Qwen3TTSTokenizer

    print(
        f"Loading Qwen3TTSTokenizer from {tokenizer_model_path} on {device}..."
    )
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
    )

    encoded: List[dict] = []
    batch_rows: List[dict] = []
    batch_audios: List[str] = []
    started = time.time()

    def _flush():
        if not batch_rows:
            return
        result = tokenizer.encode(batch_audios)
        for code, row in zip(result.audio_codes, batch_rows):
            row["audio_codes"] = code.cpu().tolist()
            encoded.append(row)
        batch_rows.clear()
        batch_audios.clear()

    for row in rows:
        batch_rows.append(row)
        batch_audios.append(row["audio"])
        if len(batch_rows) >= BATCH_INFER_NUM:
            _flush()
    _flush()

    elapsed = max(time.time() - started, 1e-3)
    print(
        f"Tokenized {len(encoded)} clips in {elapsed:.1f}s "
        f"({len(encoded) / elapsed:.1f} clips/sec)."
    )
    return encoded


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download a HuggingFace TTS dataset and produce a Qwen3-TTS "
            "training JSONL with precomputed audio_codes."
        )
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        required=True,
        help="HuggingFace dataset repo id (e.g. `org/my-tts-dataset`).",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="train.jsonl",
        help="Path to the final training JSONL with audio_codes (default: train.jsonl).",
    )
    parser.add_argument("--cache_dir", type=str, default="./hf_dataset_cache")
    parser.add_argument(
        "--ref_index",
        type=int,
        default=0,
        help="Index of the sample to use as shared reference audio (default: 0).",
    )
    parser.add_argument("--max_samples", type=int, default=None)
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
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help=(
            "Name of the column containing the transcript. Some datasets "
            "expose a normalized variant (e.g. LJ Speech has both `text` and "
            "`normalized_text`); using the normalized form generally matches "
            "the audio better (default: text)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for the Qwen3 TTS tokenizer (default: cuda:0).",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    )
    parser.add_argument(
        "--skip_tokenize",
        action="store_true",
        help=(
            "Skip the audio_codes precomputation step. Useful for quickly "
            "inspecting the downloaded manifest without loading the tokenizer."
        ),
    )
    args = parser.parse_args()

    _ensure_hf_transfer_or_warn()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows, ref_audio_path = download_dataset(
        dataset_repo=args.dataset_repo,
        cache_dir=cache_dir,
        token=args.hf_token,
        max_workers=args.max_workers,
        source=args.source,
        max_samples=args.max_samples,
        ref_index=args.ref_index,
        text_column=args.text_column,
    )

    if not args.skip_tokenize:
        rows = tokenize_audio_codes(
            rows=rows,
            tokenizer_model_path=args.tokenizer_model_path,
            device=args.device,
        )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} rows to {output_path}")
    print(f"  - Reference audio (shared by every row): {ref_audio_path}")
    if args.skip_tokenize:
        print("  - audio_codes NOT included (--skip_tokenize was set)")


if __name__ == "__main__":
    main()
