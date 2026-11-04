"""Verify and deactivate only the exact run recorded in a selected output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_env
from lifecycle import cleanup_owned_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
        help="This controller's output directory containing loops_run.json.")
    parser.add_argument("--env-file", help="Explicit credential file; never printed.")
    parser.add_argument("--yes", action="store_true", help="Confirm cleanup of this recorded owned run.")
    args = parser.parse_args()
    if not args.yes:
        parser.error("Inspect loops_run.json and pass --yes to confirm exact owned-run cleanup.")
    try:
        record = json.loads((args.output / "loops_run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        parser.error("Cannot read the selected output's loops_run.json.")
    if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
        parser.error("loops_run.json must contain the controller's exact run_id.")
    load_env(args.env_file)
    report = cleanup_owned_run(record["run_id"], args.output)
    print(json.dumps(report, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
