"""Prepare only the three SHA256-pinned tokenizer files, without model weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from common import ROOT, _json_file, load_env, sha256_file


TOKENIZER_FILES = frozenset({"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"})


def _manifest(root):
    snapshot = _json_file(root / "data_manifest.json", "data manifest").get("tokenizer", {})
    if not isinstance(snapshot, dict):
        raise ValueError("Tokenizer manifest entry must be an object.")
    hashes = snapshot.get("file_sha256")
    if snapshot.get("path") != "tokenizer" or not isinstance(hashes, dict) or set(hashes) != TOKENIZER_FILES:
        raise ValueError("Manifest must pin exactly the three packaged tokenizer filenames.")
    if any(not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
           for value in hashes.values()):
        raise ValueError("Tokenizer manifest contains an invalid SHA256 hash.")
    return hashes


def _check_cache(target, hashes):
    if target.is_symlink():
        raise ValueError("Tokenizer destination must not be a symlink.")
    if not target.exists():
        return set()
    if not target.is_dir() or any(path.name not in hashes for path in target.iterdir()):
        raise ValueError("Tokenizer destination contains unexpected entries.")
    found = set()
    for name in hashes:
        path = target / name
        if path.is_symlink():
            raise ValueError("Tokenizer destination files must not be symlinks.")
        if path.exists():
            if not path.is_file() or sha256_file(path) != hashes[name]:
                raise ValueError(f"Existing tokenizer file differs from its pinned hash: {name}")
            found.add(name)
    return found


def _download(repository, revision, directory):
    from huggingface_hub import hf_hub_download

    try:
        for name in sorted(TOKENIZER_FILES):
            hf_hub_download(repository, filename=name, repo_type="model", revision=revision,
                local_dir=directory, token=os.environ.get("HF_TOKEN") or False,
                endpoint="https://huggingface.co")
    except Exception:
        # Hub exceptions can contain signed URLs or response bodies.
        raise ValueError("Cannot download pinned tokenizer files; check repository, revision, access, and HF_TOKEN.") from None


def _write_source(root, source):
    path = root / "tokenizer_source.json"
    if path.is_symlink():
        raise ValueError("Tokenizer provenance destination must not be a symlink.")
    content = json.dumps(source, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError:
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise ValueError("Existing tokenizer provenance differs; it was not overwritten.") from None


def prepare_tokenizer(root=ROOT, *, tokenizer_dir=None, model_repository=None, revision=None,
                      record_source=False):
    """Stage and verify every file before no-clobber publication.

    Matching cached files are reused. Publication is atomic per file; an I/O
    failure may leave a verified partial cache for the next invocation.
    Local imports never initialize Hugging Face or access the network.
    """
    root = Path(root).resolve(strict=True)
    hashes = _manifest(root)
    if tokenizer_dir is None and model_repository is None and revision is None:
        snapshot = _json_file(root / "data_manifest.json", "data manifest")["tokenizer"]
        model_repository = snapshot.get("model_repository")
        revision = snapshot.get("revision")
    if tokenizer_dir is not None and (model_repository is not None or revision is not None):
        raise ValueError("Choose a local tokenizer directory or a repository/revision, not both.")
    if model_repository is not None or revision is not None:
        if (not isinstance(model_repository, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", model_repository) is None):
            raise ValueError("Model repository must be a Hugging Face owner/name identifier.")
        if not isinstance(revision, str) or re.fullmatch(r"[a-fA-F0-9]{40}", revision) is None:
            raise ValueError("Network setup requires an immutable full 40-character commit revision.")
    target = root / "tokenizer"
    cached = _check_cache(target, hashes)
    if cached == TOKENIZER_FILES:
        if record_source:
            metadata = root / "tokenizer_source.json"
            if metadata.is_symlink():
                raise ValueError("Tokenizer provenance destination must not be a symlink.")
            if not metadata.exists():
                _write_source(root, {"source": "verified-cache", "file_sha256": hashes})
        return {"status": "reused", "tokenizer_directory": "tokenizer", "file_sha256": hashes,
                "network_used": False, "compute_provisioned": False}
    if tokenizer_dir is None and model_repository is None:
        raise ValueError("Tokenizer is not prepared and its public repository/revision is not pinned in the manifest.")
    network_used = tokenizer_dir is None
    with tempfile.TemporaryDirectory(prefix=".tokenizer-setup-", dir=root) as temporary:
        scratch = Path(temporary)
        source = Path(tokenizer_dir) if tokenizer_dir is not None else scratch / "download"
        if network_used:
            _download(model_repository, revision, source)
        prepared = scratch / "verified"
        prepared.mkdir()
        for name in sorted(TOKENIZER_FILES):
            try:
                shutil.copyfile(source / name, prepared / name)
            except OSError:
                raise ValueError(f"Cannot read required tokenizer file: {name}") from None
            if sha256_file(prepared / name) != hashes[name]:
                raise ValueError(f"Tokenizer checksum mismatch: {name}; no files were published.")
        _check_cache(target, hashes)
        target.mkdir(exist_ok=True)
        for name in sorted(TOKENIZER_FILES):
            if target.is_symlink():
                raise ValueError("Tokenizer destination became a symlink.")
            try:
                os.link(prepared / name, target / name)
            except FileExistsError:
                _check_cache(target, hashes)
        _check_cache(target, hashes)
    if record_source:
        provenance = {"source": "huggingface" if network_used else "local-import", "file_sha256": hashes}
        if network_used:
            provenance.update({"model_repository": model_repository, "revision": revision.lower()})
        _write_source(root, provenance)
    return {"status": "prepared", "tokenizer_directory": "tokenizer", "file_sha256": hashes,
            "network_used": network_used, "compute_provisioned": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, help="Optional offline import; not needed for normal public setup.")
    parser.add_argument("--model-repository", help="Optional Hugging Face repository override; requires --revision.")
    parser.add_argument("--revision", help="Immutable full 40-character commit; defaults to the packaged public snapshot.")
    parser.add_argument("--env-file", help="Explicit allowlisted settings file; HF_TOKEN is optional for public files.")
    parser.add_argument("--record-source", action="store_true", help="Write optional ignored tokenizer_source.json.")
    args = parser.parse_args(argv)
    try:
        load_env(args.env_file)
        report = prepare_tokenizer(tokenizer_dir=args.tokenizer_dir, model_repository=args.model_repository,
                                   revision=args.revision, record_source=args.record_source)
    except (OSError, ValueError) as error:
        print(f"Tokenizer setup failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
