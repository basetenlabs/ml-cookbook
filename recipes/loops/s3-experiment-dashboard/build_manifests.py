#!/usr/bin/env python3
"""Index the synced experiment store: write per-run run.json + top-level
experiments.json.

This script plays the role the skill's detection phases (1-5) play: for each
run dir under ./runs it detects the metrics file (metrics.jsonl), the config
file (hyperparams.json), merges meta.json, and writes a run.json following the
rollout-dashboard schema (SKILL.md Phase 4) plus the s3-experiment-dashboard
extension keys: source, s3_uri, meta. Then it writes experiments.json — the
cross-run index the compare page is built from.

Cheap to re-run after every `sync.sh` — manifests are simply overwritten.

Usage:  S3_URI=s3://<bucket>/<prefix> python3 build_manifests.py
S3_URI is display-only provenance (it never touches AWS); omit it and the
s3_uri fields are simply left out.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs"

# Same value your team gave you for sync.sh (see README "Before you start").
S3_PREFIX = os.environ.get("S3_URI", "").rstrip("/") + "/" if os.environ.get("S3_URI") else None

EVAL_SERIES = "eval_loss"
TRAIN_SERIES = "train_loss"


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def detect_run(run_dir: Path):
    """Phases 1-4 for this store's fast path. Returns (run_json, summary)."""
    metrics_path = run_dir / "metrics.jsonl"
    hparams_path = run_dir / "hyperparams.json"
    meta_path = run_dir / "meta.json"

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    hparams = json.loads(hparams_path.read_text()) if hparams_path.exists() else {}

    records = read_jsonl(metrics_path) if metrics_path.exists() else []
    # series = every numeric key seen, minus the x axis and wall clock
    series, seen = [], set()
    for r in records:
        for k, v in r.items():
            if k in ("step", "time") or k in seen:
                continue
            if isinstance(v, (int, float)):
                seen.add(k)
                series.append(k)

    mtimes = [p.stat().st_mtime for p in run_dir.iterdir() if p.is_file()]
    last_updated = iso(max(mtimes)) if mtimes else None

    run_json = {
        "run_id": run_dir.name,
        "started_at": meta.get("started_at") or (iso(min(mtimes)) if mtimes else None),
        "last_updated": last_updated,
        "rollout_sources": [],  # this store logs metrics only — no rollout files
        "metrics": {
            "file": "metrics.jsonl",
            "x_axis": "step",
            "series": series,
        } if records else None,
        "config": {
            "file": "hyperparams.json",
            "values": flatten(hparams),
        } if hparams else None,
        # --- s3-experiment-dashboard extension keys ---
        "source": meta.get("source"),
        "s3_uri": S3_PREFIX + run_dir.name + "/" if S3_PREFIX else None,
        "meta": {k: v for k, v in meta.items() if k not in ("source", "started_at")},
    }
    run_json = {k: v for k, v in run_json.items() if v is not None}

    # summary stats for experiments.json
    evals = [(r["step"], r[EVAL_SERIES]) for r in records if EVAL_SERIES in r]
    trains = [(r["step"], r[TRAIN_SERIES]) for r in records if TRAIN_SERIES in r]
    summary = {
        "final_eval_loss": round(evals[-1][1], 4) if evals else None,
        "best_eval_loss": round(min(v for _, v in evals), 4) if evals else None,
        "final_train_loss": round(trains[-1][1], 4) if trains else None,
        "last_step": records[-1]["step"] if records else None,
    }
    return run_json, summary, flatten(hparams)


def flatten(d, prefix=""):
    """Flatten nested config to scalars; nested keys dotted (optimizer.lr)."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def main():
    if not RUNS_ROOT.is_dir():
        raise SystemExit(f"{RUNS_ROOT} not found — run ./sync.sh first "
                         "(or example/generate_runs.py for the demo)")
    runs = []
    for run_dir in sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir()):
        run_json, summary, hparams = detect_run(run_dir)
        (run_dir / "run.json").write_text(json.dumps(run_json, indent=2) + "\n")
        runs.append({
            "run_id": run_json["run_id"],
            "path": f"runs/{run_dir.name}",
            "source": run_json.get("source"),
            "started_at": run_json.get("started_at"),
            "last_updated": run_json.get("last_updated"),
            # every numeric key the run logs — the compare page's metric
            # selector is the union of these across runs
            "series": (run_json.get("metrics") or {}).get("series", []),
            "hparams": hparams,
            "summary": summary,
        })
        print(f"indexed {run_dir.name}: last_step={summary['last_step']} "
              f"best_eval={summary['best_eval_loss']}")

    experiments = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **({"s3_prefix": S3_PREFIX} if S3_PREFIX else {}),
        "runs": runs,
    }
    (ROOT / "experiments.json").write_text(json.dumps(experiments, indent=2) + "\n")
    print(f"wrote experiments.json with {len(runs)} runs")


if __name__ == "__main__":
    main()
