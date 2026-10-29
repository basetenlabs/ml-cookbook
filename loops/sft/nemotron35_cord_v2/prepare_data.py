import argparse
import json
import os
import uuid

from huggingface_hub import HfApi, hf_hub_download
from pyarrow import parquet
from transformers import AutoTokenizer

from common import ROOT, dataset_cache, settings
from data import SPLIT_ROWS, load_prepared, prepare_row, sha256
from prepare_assets import verify_assets


def main():
    parser = argparse.ArgumentParser(
        description="Download and audit CORD-v2 train/validation without provisioning GPUs."
    )
    parser.add_argument("--max-seq-len", type=int)
    arguments = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    config = settings()
    asset_audit = verify_assets()
    if arguments.max_seq_len is not None:
        if arguments.max_seq_len < 1:
            parser.error("--max-seq-len must be positive.")
        config["max_seq_len"] = arguments.max_seq_len
    destination = ROOT / "data" / "prepared"
    if destination.exists():
        _, audit = load_prepared(config)
        print(json.dumps({"cached": True, "audit": audit}, indent=2))
        return
    staging = destination.with_name("preparing-" + uuid.uuid4().hex)
    (staging / "images").mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "assets" / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    info = HfApi().dataset_info(
        config["dataset"], revision=config["dataset_revision"], files_metadata=True
    )
    if info.sha != config["dataset_revision"]:
        raise ValueError("Dataset revision did not resolve to the pinned commit.")
    audit = {
        "dataset": config["dataset"],
        "dataset_revision": info.sha,
        "max_seq_len": config["max_seq_len"],
        "splits": {},
        "truncated_rows": 0,
        "test_split_used": False,
        "masking": config["managed_settings"]["masking"],
        "image_pixels": config["managed_settings"]["image_pixels"],
        "tokenizer_sha256": sha256(ROOT / "assets" / "tokenizer" / "tokenizer.json"),
        "processor_sha256": sha256(ROOT / "assets" / "preprocessor_config.json"),
        "asset_audit": asset_audit,
    }
    original_hashes = {}
    for split, expected_count in SPLIT_ROWS.items():
        filenames = sorted(
            entry.rfilename
            for entry in info.siblings
            if entry.rfilename.startswith(f"data/{split}-")
            and entry.rfilename.endswith(".parquet")
        )
        if not filenames:
            raise ValueError(f"No parquet files for the {split} split.")
        encoded_rows = []
        source_files = []
        for filename in filenames:
            print(f"Downloading {split}: {filename}", flush=True)
            path = hf_hub_download(
                config["dataset"],
                filename,
                repo_type="dataset",
                revision=info.sha,
                cache_dir=dataset_cache(),
            )
            source_files.append({"file": filename, "sha256": sha256(path)})
            for batch in parquet.ParquetFile(path).iter_batches(
                batch_size=1, columns=["image", "ground_truth"]
            ):
                for row in batch.to_pylist():
                    try:
                        encoded_rows.append(
                            prepare_row(
                                row,
                                len(encoded_rows),
                                split,
                                tokenizer,
                                config,
                                staging,
                            )
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"{split} row {len(encoded_rows)}: {error}; partial preparation retained in {staging}"
                        ) from error
                    if len(encoded_rows) % 50 == 0:
                        print(
                            f"Audited {split}: {len(encoded_rows)}/{expected_count}",
                            flush=True,
                        )
        if len(encoded_rows) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {split} rows, found {len(encoded_rows)}."
            )
        original_hashes[split] = {row["original_image_sha256"] for row in encoded_rows}
        output = staging / f"{split}.jsonl"
        with output.open("x") as handle:
            for row in encoded_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        audit["splits"][split] = {
            "rows": len(encoded_rows),
            "sha256": sha256(output),
            "source_files": source_files,
            "maximum_tokens": max(row["sequence_tokens"] for row in encoded_rows),
            "active_tokens": sum(row["active_tokens"] for row in encoded_rows),
            "resized_images": sum(
                not row["exact_image_pixels"] for row in encoded_rows
            ),
        }
    if original_hashes["train"] & original_hashes["validation"]:
        raise ValueError("Official training and validation image bytes overlap.")
    (staging / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    os.rename(staging, destination)
    _, verified = load_prepared(config)
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
