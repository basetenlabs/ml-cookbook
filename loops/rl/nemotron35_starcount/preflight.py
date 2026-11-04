"""Audit the frozen star-count Loops recipe without provisioning resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from common import load_env, prepare, settings, validate_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Existing key file; reads only Baseten and W&B credentials.")
    parser.add_argument("--offline", action="store_true", help="Skip authentication and all API requests.")
    parser.add_argument("--output", type=Path, help="Optional JSON audit output path.")
    parser.add_argument("--verbose", action="store_true", help="Print the full image-dimension audit.")
    arguments = parser.parse_args(argv)
    try:
        if not arguments.offline:
            load_env(arguments.env_file)
        config = settings()
        capability = None
        if not arguments.offline:
            capability = validate_catalog(config)
        data, audit = prepare(config)
        del data
        audit["offline"] = arguments.offline
        audit["capability_checked"] = capability is not None
        if capability is not None:
            audit["loops_model_capability"] = capability
        if arguments.output:
            arguments.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        displayed = audit if arguments.verbose else {
            **audit, "splits": {split: {key: value for key, value in entry.items()
                if key != "original_image_dimensions"} for split, entry in audit.get("splits", {}).items()}}
        print(json.dumps(displayed, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
