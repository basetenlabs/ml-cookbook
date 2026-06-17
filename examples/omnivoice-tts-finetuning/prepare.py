#!/usr/bin/env python3
"""
Download a single-speaker TTS dataset from Hugging Face, materialize the wav
clips locally, and write OmniVoice-style training/dev JSONL manifests.

Unlike the Qwen3-TTS recipe, OmniVoice tokenizes audio in a separate stage
(`omnivoice.scripts.extract_audio_tokens`), so this script does NOT precompute
audio tokens. It only produces the raw manifests that the tokenizer consumes.

Pipeline:
  HF dataset  ->  local wav files  ->  {train,dev}.jsonl

Output JSONL fields (one JSON object per line), matching the format expected by
OmniVoice's fine-tuning pipeline:
- id:           unique sample id (derived from the clip file name / dataset id)
- audio_path:   absolute path to the wav clip
- text:         transcript corresponding to audio
- language_id:  language code for the clip (e.g. `en`), constant for LJ Speech

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
import random
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

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


def _id_from_file_name(file_name: str) -> str:
    """Derive a stable sample id from a clip path (drop dirs + extension)."""
    return Path(file_name).stem


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
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    Materialize the embedded audio bytes from the parquet shards as local wav
    files.

    Two-pass to avoid extracting wavs we'd immediately throw away:
      Pass 1: metadata-only scan (skips the heavy `audio` bytes column) to
              build candidate rows, sort them by file_name, and apply
              ``max_samples`` if set.
      Pass 2: for each shard that still contributes a surviving row, read the
              `audio` column and write only those wavs to disk.
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

    # ---- Pass 1: metadata-only scan ----
    # We project just the columns needed to compute `file_name` and pick up
    # `text`. Crucially we skip the `audio` column so we don't pull GBs of
    # inline wav bytes through memory just to discard most of them.
    candidates: List[dict] = []
    for shard_idx, pq_file in enumerate(parquet_files):
        col_names = pq.read_schema(pq_file).names

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

        meta_cols = [text_column]
        for opt in ("id", "file_name"):
            if opt in col_names and opt not in meta_cols:
                meta_cols.append(opt)
        meta_rows = pq.read_table(pq_file, columns=meta_cols).to_pylist()

        for row_idx, record in enumerate(meta_rows):
            text = record.get(text_column)
            if not text:
                continue

            row_id = record.get("id")
            if record.get("file_name"):
                file_name = record["file_name"]
            elif row_id:
                file_name = f"clips/{row_id}.wav"
            else:
                # Deterministic fallback; ordered by (shard, in-shard index)
                # so it's stable across runs.
                file_name = f"clips/{shard_idx:04d}_{row_idx:06d}.wav"

            candidates.append({
                "shard_path": pq_file,
                "row_idx": row_idx,
                "file_name": file_name,
                "text": text,
            })

    total_candidates = len(candidates)
    candidates.sort(key=lambda c: c["file_name"])
    if max_samples is not None and max_samples > 0:
        candidates = candidates[:max_samples]
    print(
        f"Parquet metadata scan: {total_candidates} candidate rows"
        + (
            f" (selecting first {len(candidates)} after sort, "
            f"max_samples={max_samples})"
            if len(candidates) != total_candidates
            else ""
        )
    )

    # Group surviving rows by shard so each shard is read at most once.
    by_shard: "dict[Path, List[dict]]" = {}
    for c in candidates:
        by_shard.setdefault(c["shard_path"], []).append(c)

    # ---- Pass 2: write only the wavs we actually keep ----
    rows: List[dict] = []
    written = 0
    started = time.time()
    for pq_file, shard_cands in by_shard.items():
        # If every selected wav for this shard already exists on disk (re-run
        # of prepare.py with the same --max_samples), skip the parquet read
        # entirely.
        pending = [
            c for c in shard_cands
            if not (clips_dir / c["file_name"]).resolve().exists()
        ]
        audio_col = None
        if pending:
            audio_col = pq.read_table(pq_file, columns=["audio"])["audio"]

        for c in shard_cands:
            file_name = c["file_name"]
            out_path = (clips_dir / file_name).resolve()
            if not out_path.exists():
                audio = audio_col[c["row_idx"]].as_py()
                if isinstance(audio, dict):
                    audio_bytes = audio.get("bytes")
                else:
                    audio_bytes = audio
                if not audio_bytes:
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                written += 1

            rows.append({
                "file_name": file_name,
                "audio_path": str(out_path),
                "text": c["text"],
            })

    elapsed = max(time.time() - started, 1e-3)
    rate = written / elapsed if written else 0.0
    print(
        f"Extracted {len(rows)} rows ({written} new wav files written) "
        f"in {elapsed:.1f}s ({rate:.0f} files/sec written)."
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
    max_samples: Optional[int] = None,
) -> Tuple[Path, List[dict]]:
    """
    Two-pass AudioFolder fetch:
      Pass 1: pull just metadata (manifests, READMEs) so we can enumerate
              every candidate row without downloading any wavs.
      Pass 2: sort + apply ``max_samples``, then targeted-download only the
              wavs for the surviving rows.
    """
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

    total_candidates = len(rows)
    rows.sort(key=lambda r: r["file_name"])
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    if rows and len(rows) != total_candidates:
        print(
            f"AudioFolder metadata scan: {total_candidates} candidate rows "
            f"(selecting first {len(rows)} after sort, max_samples={max_samples})."
        )

    missing_files = sorted({
        r["file_name"] for r in rows if not Path(r["audio_path"]).exists()
    })
    if missing_files:
        print(
            f"Downloading {len(missing_files)} audio clips with "
            f"max_workers={max_workers}..."
        )
        _download_clips(
            dataset_repo=dataset_repo,
            cache_dir=cache_dir,
            file_names=missing_files,
            token=token,
            max_workers=max_workers,
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
    text_column: str,
    language_id: str,
) -> List[dict]:
    """
    Materialize wav clips on disk and return a list of OmniVoice manifest rows:
      {id, audio_path, text, language_id}
    (paths absolute).
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
                max_samples=max_samples,
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
            max_samples=max_samples,
        )

    if not rows:
        raise SystemExit("No usable rows found in the dataset.")

    print(f"Samples: {len(rows)}")

    still_missing = [r for r in rows if not Path(r["audio_path"]).exists()]
    if still_missing:
        raise SystemExit(
            f"Expected audio file(s) missing after download, e.g. "
            f"{still_missing[0]['file_name']!r}. Check --hf_token and dataset access."
        )

    print(f"Dataset cache: {repo_root}")

    return [
        {
            "id": _id_from_file_name(row["file_name"]),
            "audio_path": row["audio_path"],
            "text": row["text"],
            "language_id": language_id,
        }
        for row in rows
    ]


# ---------- manifest writing ----------


def _write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_dev(
    rows: List[dict],
    dev_size: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Deterministically hold out ``dev_size`` rows for validation.

    We shuffle indices with a seeded RNG (rather than slicing the head/tail)
    because the rows are produced by a sorted pipeline (sorted by file_name),
    and a contiguous block would skew the dev set toward one end of the corpus.
    """
    if dev_size <= 0:
        return rows, []
    if dev_size >= len(rows):
        raise SystemExit(
            f"--dev_size {dev_size} must be smaller than the dataset "
            f"({len(rows)} rows)."
        )
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    dev_idx = set(indices[:dev_size])
    dev_rows = [rows[i] for i in sorted(dev_idx)]
    train_rows = [row for i, row in enumerate(rows) if i not in dev_idx]
    return train_rows, dev_rows


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download a HuggingFace TTS dataset and produce OmniVoice "
            "fine-tuning JSONL manifests (train + dev)."
        )
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        required=True,
        help="HuggingFace dataset repo id (e.g. `SeanSleat/lj_speech`).",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="data/finetune/manifests/train.jsonl",
        help="Path to the output training JSONL manifest.",
    )
    parser.add_argument(
        "--dev_jsonl",
        type=str,
        default="data/finetune/manifests/dev.jsonl",
        help="Path to the output dev JSONL manifest (skipped if --dev_size 0).",
    )
    parser.add_argument("--cache_dir", type=str, default="./hf_dataset_cache")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap on the total number of clips before the train/dev split.",
    )
    parser.add_argument(
        "--dev_size",
        type=int,
        default=50,
        help="Number of rows held out (deterministically) for the dev set. "
        "Set to 0 to skip the dev manifest.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="RNG seed for the deterministic train/dev split.",
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
        "--language_id",
        type=str,
        default="en",
        help=(
            "Language code written to every manifest row's `language_id` "
            "field (LJ Speech is English -> `en`). OmniVoice treats this as "
            "optional metadata (default: en)."
        ),
    )
    args = parser.parse_args()

    _ensure_hf_transfer_or_warn()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = download_dataset(
        dataset_repo=args.dataset_repo,
        cache_dir=cache_dir,
        token=args.hf_token,
        max_workers=args.max_workers,
        source=args.source,
        max_samples=args.max_samples,
        text_column=args.text_column,
        language_id=args.language_id,
    )

    train_rows, dev_rows = split_train_dev(
        rows, dev_size=args.dev_size, seed=args.split_seed
    )

    train_path = Path(args.train_jsonl)
    _write_jsonl(train_rows, train_path)
    print(f"\nWrote {len(train_rows)} rows to {train_path}")

    if dev_rows:
        dev_path = Path(args.dev_jsonl)
        _write_jsonl(dev_rows, dev_path)
        print(f"Wrote {len(dev_rows)} rows to {dev_path}")
    else:
        print("Dev manifest skipped (--dev_size 0).")


if __name__ == "__main__":
    main()
