#!/usr/bin/env python3
"""Validate .examples-index.json, the file `truss train init --examples` reads.

Fails if an indexed path is missing or has no config*.py, or if a directory
under training-jobs/examples/ with a config*.py is not indexed.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / ".examples-index.json"
EXAMPLES_DIR = ROOT / "training-jobs" / "examples"


def has_config(path: Path) -> bool:
    return any(path.glob("config*.py"))


def main() -> int:
    index = json.loads(INDEX.read_text())
    errors = []

    if index.get("version") != 1:
        errors.append(f"unsupported index version: {index.get('version')!r}")

    examples = index.get("examples", {})
    indexed_paths = set()
    for name, entry in examples.items():
        rel = entry.get("path")
        if not rel:
            errors.append(f"{name}: missing 'path'")
            continue
        path = ROOT / rel
        indexed_paths.add(path.resolve())
        if not path.is_dir():
            errors.append(f"{name}: path does not exist: {rel}")
        elif not has_config(path):
            errors.append(f"{name}: no config*.py in {rel}")

    for example in sorted(p for p in EXAMPLES_DIR.iterdir() if p.is_dir()):
        candidates = [example / "training", example]
        trainable = [c for c in candidates if c.is_dir() and has_config(c)]
        if trainable and not any(c.resolve() in indexed_paths for c in trainable):
            errors.append(f"not indexed: {trainable[0].relative_to(ROOT)}")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} problem(s) in {INDEX.name}", file=sys.stderr)
        return 1
    print(f"{INDEX.name}: {len(examples)} examples OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
