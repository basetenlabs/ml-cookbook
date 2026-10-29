import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from common import ROOT, settings
from vision import load_processor_config

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
REQUIRED_FILES = {
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "preprocessor_config.json",
}
ALLOWED_FILES = REQUIRED_FILES | {"tokenizer/chat_template.jinja"}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_matches(files):
    expected = json.loads((ROOT / "reference_fingerprints.json").read_text())
    return all(files.get(name) == digest for name, digest in expected.items())


def verify_assets(directory=None):
    directory = ROOT / "assets" if directory is None else Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except FileNotFoundError:
        raise ValueError(
            "Run uv run python prepare_assets.py first; model assets are not bundled."
        ) from None
    files = manifest["files"]
    if not REQUIRED_FILES.issubset(files) or not set(files).issubset(ALLOWED_FILES):
        raise ValueError("Asset manifest contains missing or unexpected files.")
    for name, expected in files.items():
        if fingerprint(directory / name) != expected:
            raise ValueError(
                f"Asset fingerprint changed: {name}. Existing prepared rows must not be reused."
            )
    load_processor_config(directory / "preprocessor_config.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Automatically fetch tokenizer/processor metadata from Hugging Face, never model weights or GPUs."
    )
    parser.add_argument(
        "--model-repository",
        default=settings()["base_model"],
        help="Hugging Face model repository; defaults to the recipe's base model.",
    )
    parser.add_argument("--revision", help="Optional immutable 40-character HF commit.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Optional offline/historical override; not needed for normal setup.",
    )
    parser.add_argument(
        "--processor-config",
        type=Path,
        help="Local processor config, required only with --tokenizer-dir.",
    )
    parser.add_argument("--require-reference-fingerprints", action="store_true")
    arguments = parser.parse_args()
    if arguments.revision is not None and not re.fullmatch(
        r"[0-9a-f]{40}", arguments.revision
    ):
        parser.error(
            "--revision must be an immutable 40-character commit, not a moving branch or tag."
        )
    if bool(arguments.tokenizer_dir) != bool(arguments.processor_config):
        parser.error(
            "Local import requires both --tokenizer-dir and --processor-config."
        )
    if arguments.tokenizer_dir is not None and arguments.revision is not None:
        parser.error("--revision applies to HF downloads, not local imports.")
    destination = ROOT / "assets"
    if destination.exists():
        manifest = verify_assets(destination)
        if (
            arguments.revision is not None
            and manifest["source"].get("revision") != arguments.revision
        ):
            raise ValueError(
                "Cached asset revision differs; use a fresh recipe copy without overwriting the cache."
            )
        if arguments.require_reference_fingerprints and not reference_matches(
            manifest["files"]
        ):
            raise ValueError(
                "Cached assets differ from the historical run; leave them intact and use a fresh recipe copy."
            )
        print(json.dumps({"cached": True, "audit": manifest}, indent=2))
        return
    staging = ROOT / ("assets-preparing-" + uuid.uuid4().hex)
    (staging / "tokenizer").mkdir(parents=True)
    if arguments.tokenizer_dir is not None:
        for name in TOKENIZER_FILES:
            source = arguments.tokenizer_dir / name
            if source.is_file():
                shutil.copyfile(source, staging / "tokenizer" / name)
        shutil.copyfile(
            arguments.processor_config, staging / "preprocessor_config.json"
        )
        origin = {"kind": "local_import", "revision": None}
    else:
        try:
            info = HfApi().model_info(
                arguments.model_repository, revision=arguments.revision
            )
            names = {entry.rfilename for entry in info.siblings}
            wanted = {name for name in TOKENIZER_FILES if name in names} | {
                "preprocessor_config.json"
            }
            if not {
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
            }.issubset(names):
                raise ValueError(
                    "Repository lacks required metadata; supply an authorized local tokenizer and processor config."
                )
            for name in sorted(wanted):
                source = hf_hub_download(
                    arguments.model_repository, name, revision=info.sha
                )
                target = (
                    staging / name
                    if name == "preprocessor_config.json"
                    else staging / "tokenizer" / name
                )
                shutil.copyfile(source, target)
            origin = {
                "kind": "huggingface",
                "repository": arguments.model_repository,
                "revision": info.sha,
            }
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                "Tokenizer/processor metadata could not be downloaded. Check connectivity "
                "and any configured HF credentials, or use the optional offline overrides. "
                "No trainer was allocated."
            ) from error
    files = {
        str(path.relative_to(staging)): fingerprint(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    if not REQUIRED_FILES.issubset(files):
        raise ValueError(
            "Local tokenizer must contain tokenizer.json and tokenizer_config.json."
        )
    processor = load_processor_config(staging / "preprocessor_config.json")
    tokenizer = AutoTokenizer.from_pretrained(
        staging / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError(
            "A fast tokenizer with the Nemotron vision chat template is required."
        )
    from data import encode_row

    downsample_factor = round(1 / processor["downsample_ratio"])
    image_tokens = processor["min_num_patches"] // downsample_factor**2
    encode_row(tokenizer, "<s_total>42</s_total>", image_tokens, 8192)
    matches = reference_matches(files)
    if arguments.require_reference_fingerprints and not matches:
        raise ValueError(
            "Assets do not match the historical fingerprints; partial import is retained but no GPUs were allocated."
        )
    manifest = {
        "source": origin,
        "files": files,
        "historical_fingerprints_match": matches,
        "managed_processor_identity_verified": False,
        "remote_code_executed": False,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_assets(staging)
    os.rename(staging, destination)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
