"""Copy a supported Whisper checkpoint into the Training checkpoint directory."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

COMMON_SHAPE = {
    "model_type": "whisper",
    "architectures": ["WhisperForConditionalGeneration"],
    "d_model": 1280,
    "encoder_layers": 32,
    "encoder_attention_heads": 20,
    "decoder_attention_heads": 20,
    "encoder_ffn_dim": 5120,
    "decoder_ffn_dim": 5120,
    "max_source_positions": 1500,
    "max_target_positions": 448,
}
MODEL_SHAPES = {
    "large-v2": {"decoder_layers": 32, "num_mel_bins": 80, "vocab_size": 51865},
    "large-v3": {"decoder_layers": 32, "num_mel_bins": 128, "vocab_size": 51866},
    "large-v3-turbo": {"decoder_layers": 4, "num_mel_bins": 128, "vocab_size": 51866},
}
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")
SAFETENSORS_INDEX = "model.safetensors.index.json"
METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "vocab.json",
    "merges.txt",
)


def download_checkpoint(source: Path) -> str:
    """Resolve once, then download only checkpoint files from the immutable commit."""
    repo = os.environ["SOURCE_REPO"]
    token = os.environ.get("HF_TOKEN") or False
    info = HfApi().model_info(repo, revision=os.environ["SOURCE_REVISION"], token=token)
    commit = info.sha
    if not commit:
        raise ValueError("Hugging Face did not return a checkpoint commit")
    print(f"Downloading {repo}@{commit}", flush=True)
    start_sec = time.monotonic()
    available = {entry.rfilename for entry in info.siblings}
    hf_hub_download(repo, "config.json", revision=commit, token=token, local_dir=source)
    checkpoint_model(json.loads((source / "config.json").read_text()))
    if "model.safetensors" in available:
        weights = ["model.safetensors"]
    elif SAFETENSORS_INDEX in available:
        hf_hub_download(
            repo, SAFETENSORS_INDEX, revision=commit, token=token, local_dir=source
        )
        shards = set(
            json.loads((source / SAFETENSORS_INDEX).read_text())["weight_map"].values()
        )
        invalid = sorted(
            s
            for s in shards
            if Path(s).name != s or not s.endswith(".safetensors") or s not in available
        )
        if not shards or invalid:
            raise ValueError(
                f"{SAFETENSORS_INDEX} references missing or invalid shards: {invalid}"
            )
        weights = [SAFETENSORS_INDEX, *sorted(shards)]
    else:
        raise ValueError(
            "Source has no safetensors weights; merge adapters into full weights first"
        )
    snapshot_download(
        repo,
        revision=commit,
        token=token,
        local_dir=source,
        allow_patterns=[*METADATA_FILES, *weights],
    )
    print(f"Downloaded in {time.monotonic() - start_sec:.1f}s", flush=True)
    return commit


def checkpoint_model(config: dict) -> str:
    for model, shape in MODEL_SHAPES.items():
        if all(
            config.get(key) == value for key, value in {**COMMON_SHAPE, **shape}.items()
        ):
            return model
    fields = {
        key: config.get(key)
        for key in COMMON_SHAPE.keys()
        | {"decoder_layers", "num_mel_bins", "vocab_size"}
    }
    raise ValueError(f"Unsupported Whisper checkpoint architecture: {fields}")


def main():
    with tempfile.TemporaryDirectory(prefix="whisper-import-") as directory:
        source = Path(directory)
        commit = download_checkpoint(source)
        save_checkpoint(source, commit)


def save_checkpoint(source: Path, commit: str):
    name = os.environ["CHECKPOINT_NAME"]
    # Training syncs only direct children of BT_CHECKPOINT_DIR.
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(
            f"CHECKPOINT_NAME must be a single directory name, got {name!r}"
        )
    checkpoint = Path(os.environ["BT_CHECKPOINT_DIR"]) / name
    if checkpoint.exists():
        raise FileExistsError(
            f"{checkpoint} already exists; choose another CHECKPOINT_NAME"
        )

    config = json.loads((source / "config.json").read_text())
    model = checkpoint_model(config)
    base_model = f"openai/whisper-{model}"
    files = checkpoint_files(source)

    print(f"Copying {len(files)} files to {checkpoint}", flush=True)
    start_sec = time.monotonic()
    checkpoint.mkdir(parents=True)
    for fname in files:
        if fname != "config.json":
            shutil.copyfile(source / fname, checkpoint / fname)
    (checkpoint / "import_provenance.json").write_text(
        json.dumps(
            {
                "source_repo": os.environ["SOURCE_REPO"],
                "source_revision": commit,
                "base_model": base_model,
                "optimizer_steps": 0,
            },
            indent=2,
        )
        + "\n"
    )
    # The only file not copied byte for byte: Training reads the base model from _name_or_path,
    # which a fine-tune's config may omit or point elsewhere. import_provenance.json keeps the
    # true source.
    config["_name_or_path"] = base_model
    # Training registers a checkpoint once its config.json exists, so it is written last.
    (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Copied in {time.monotonic() - start_sec:.1f}s", flush=True)
    print(
        f"Imported {base_model} as {checkpoint.name}; wait for sync and check available Training templates",
        flush=True,
    )


def checkpoint_files(source: Path) -> list[str]:
    """Files to copy, after checking the weights and tokenizer are complete."""
    if (source / "model.safetensors").is_file():
        weights = ["model.safetensors"]
    elif (source / SAFETENSORS_INDEX).is_file():
        shards = set(
            json.loads((source / SAFETENSORS_INDEX).read_text())["weight_map"].values()
        )
        invalid = sorted(
            s
            for s in shards
            if Path(s).name != s
            or not s.endswith(".safetensors")
            or not (source / s).is_file()
        )
        if not shards or invalid:
            raise ValueError(
                f"{SAFETENSORS_INDEX} references missing or invalid shards: {invalid}"
            )
        weights = [SAFETENSORS_INDEX, *sorted(shards)]
    else:
        raise ValueError("Source has no safetensors weights")

    if not (source / "tokenizer.json").is_file() and not all(
        (source / f).is_file() for f in TOKENIZER_FILES
    ):
        raise ValueError("Source tokenizer is incomplete")
    others = sorted(
        p.name
        for p in source.iterdir()
        if p.is_file() and p.suffix in (".json", ".txt") and p.name != SAFETENSORS_INDEX
    )
    return [*weights, *others]


if __name__ == "__main__":
    main()
