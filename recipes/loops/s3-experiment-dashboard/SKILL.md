---
name: s3-experiment-dashboard
description: Build a local browser dashboard over an S3-backed experiment store. Loops and Training Jobs v1 runs upload metrics.jsonl + hyperparams.json + meta.json to a shared bucket prefix; this skill syncs the prefix down, indexes the runs, and serves a local UI for loss curves, hyperparameter filtering, and cross-run comparison. S3 is the store of record; the dashboard is a read-only local viewer. Use whenever the user wants to compare training runs, view loss curves across experiments, or browse a team's S3 experiment store — even if they don't say "dashboard".
---

# S3 Experiment Dashboard

A multi-run fork of the `rollout-dashboard` skill (same folder structure, same renderer heritage). Where rollout-dashboard views one local run, this views a whole S3 prefix of runs and puts **cross-run comparison first**: which learning rate won, which run diverged, how the eval-loss curves overlay.

This skill is three things:
1. **A sync step** — `aws s3 sync` pulls the shared prefix into a local mirror. S3 stays the store of record; nothing is ever written back.
2. **An indexer** (`build_manifests.py`) — scans each synced run, writes a per-run `run.json` (the rollout-dashboard schema, plus three extension keys) and a top-level `experiments.json` cross-run index.
3. **A pre-built renderer** (same folder) — `dashboard.py`, `compare.html`, `index.html`, `detail.html`, `style.css`, `renderers.js`, `chart.umd.min.js`. Already written, already tested. Don't regenerate it; copy it.

## What you do, in order

1. Ask for the S3 URI (`s3://<bucket>/<prefix>`) and a local mirror directory — only if not given. **Never guess or fabricate bucket names or credentials** — they must come from the user (see README.md "Before you start").
2. Copy this folder's renderer files plus `build_manifests.py` and `sync.sh` into the mirror directory (not into each run dir).
3. Run `S3_URI=s3://<bucket>/<prefix> ./sync.sh`. Report run count and total size. On `AccessDenied` / missing-credential errors, **stop** and tell the user to fix `aws configure` / `AWS_PROFILE` — do not work around it.
4. Run `S3_URI=... python3 build_manifests.py`. It writes `<run_dir>/run.json` for every run plus `experiments.json` at the root.
5. Tell the user to run `python3 dashboard.py` and stop.

**Do not** edit the renderer. **Do not** re-upload anything to S3. Generated files (`run.json`, `experiments.json`, the renderer) stay local — never sync them back.

## Operating rules

- Same as rollout-dashboard: use Read/Glob/Grep to inspect, peek don't slurp, don't run or curl the dashboard yourself — hand off and stop.
- Credentials are a hard boundary: if `aws s3 ls <S3_URI>` fails, the fix is a human providing credentials, not you inventing configuration.
- The manifests are cheap to rebuild. When the user wants fresh data, re-run steps 3–4; existing manifests are overwritten and the running server picks changes up on the next browser refresh.

## The store contract (fast path)

Each run directory under the prefix is expected to contain:

- `metrics.jsonl` — one JSON object per logged step: `{"step": 120, "time": 1753...9, "train_loss": 1.83, "learning_rate": 2.4e-5, ...}` with `eval_loss` on eval steps. Every numeric key except `step`/`time` becomes a plottable series (a chart in the single-run view, an entry in the compare page's metric selector) — the loss keys are just the convention the summary stats and the selector default use.
- `hyperparams.json` — a (possibly nested) dict of hyperparameters. Nested keys get dot-flattened (`optimizer.lr`).
- `meta.json` — written once at launch: `source` (e.g. `"ec2"` or `"baseten"`), `started_at`, `git_sha`, free-form extras.

`build_manifests.py` implements detection phases 1–5 for exactly this layout. For run directories that *don't* conform (extra rollout files, other metric formats), the full detection logic in [rollout-dashboard's SKILL.md](../rollout-dashboard/SKILL.md) applies unchanged — score the files, identify the fields, and hand-edit that run's `run.json`; the renderer treats it like any other run.

### Training Jobs v1 producer

Use [`training_jobs_v1/s3_experiment_logger.py`](training_jobs_v1/s3_experiment_logger.py) inside a Training Jobs v1 project. The runtime injects `BT_TRAINING_JOB_ID`, `BT_TRAINING_PROJECT_ID`, their names, and `BT_NODE_RANK`. The adapter uses these values for run identity and metadata. It checks distributed ranks so only the primary process writes.

Map a customer-managed bucket URI and AWS credentials through the job's `Runtime.environment_variables`. Use `SecretReference` for both credential values. The complete `config.py` and training-loop snippets are in [`training_jobs_v1/README.md`](training_jobs_v1/README.md).

The adapter writes outside the Baseten checkpoint directory. Baseten manages checkpoints as model outputs, while the dashboard requires a customer-readable S3 prefix shared across Loops and Training Jobs v1 runs.

## Manifest schemas

### Per-run `run.json`

The rollout-dashboard schema, unchanged, plus three optional extension keys. Old manifests still render here, and these manifests still render in an unmodified rollout-dashboard (the renderer ignores unknown keys).

```json
{
  "run_id": "sft-llama8b-lr3e5-2026-07-28-1412",
  "started_at": "2026-07-28T14:12:03Z",
  "last_updated": "2026-07-28T19:44:10Z",
  "rollout_sources": [],
  "metrics": {"file": "metrics.jsonl", "x_axis": "step",
              "series": ["train_loss", "eval_loss", "learning_rate", "grad_norm"]},
  "config": {"file": "hyperparams.json",
             "values": {"lr": 3e-05, "batch_size": 64}},

  "source": "ec2",
  "s3_uri": "s3://bkt/exp/sft-llama8b-lr3e5-2026-07-28-1412/",
  "meta": {"git_sha": "ab12cd3", "host": "..."}
}
```

`build_manifests.py` always writes `rollout_sources: []` — it indexes metrics/config only, and the per-run view degrades to metrics-and-config, which is the primary use here. Rollout panels appear only when a run's `run.json` lists rollout sources, which means running the detection phases above (or hand-editing) for that run.

### Top-level `experiments.json`

The cross-run index the compare page is built from:

```json
{
  "generated_at": "2026-07-29T09:02:11Z",
  "s3_prefix": "s3://bkt/exp/",
  "runs": [{
    "run_id": "sft-llama8b-lr3e5-2026-07-28-1412",
    "path": "runs/sft-llama8b-lr3e5-2026-07-28-1412",
    "source": "ec2",
    "started_at": "2026-07-28T14:12:03Z",
    "last_updated": "2026-07-28T19:44:10Z",
    "series": ["train_loss", "eval_loss", "learning_rate", "grad_norm"],
    "hparams": {"lr": 3e-05, "batch_size": 64},
    "summary": {"final_eval_loss": 1.842, "best_eval_loss": 1.831,
                "final_train_loss": 1.612, "last_step": 4200}
  }]
}
```

`hparams` is the config flattened to scalars — the filter chips are built from keys whose values vary across runs. `series` is every numeric key the run logs; the compare page's metric selector is the union of these across runs. `summary` is computed at index time so the compare page loads fast regardless of metrics-file size.

## The routes (for orientation, not for editing)

- `/` and `/compare` — the compare page: run table, filter chips, overlay chart.
- `/run/<run_id>/` — the full single-run dashboard from rollout-dashboard, unchanged.
- `/experiments`, `/run/<run_id>/data`, `/run/<run_id>/detail/...` — JSON/data endpoints.

Everything is read from disk at request time; a re-sync shows up on refresh.

## Hand off

Tell the user:

```
Dashboard ready at <mirror_dir>/.

To start:
  cd <mirror_dir> && python3 dashboard.py

Then open http://localhost:8765.
  compare view: /          (all runs, filter chips, overlay chart)
  per-run view: /run/<id>/

To pull new runs later:
  S3_URI=<uri> ./sync.sh && S3_URI=<uri> python3 build_manifests.py
```

Don't run the dashboard yourself. Don't curl it. Stop.

## Reference

- `README.md` — human/agent setup doc, including what training code must write
- `sync.sh` — the S3 pull-down (read-only)
- `build_manifests.py` — writes `run.json` per run + `experiments.json`
- `dashboard.py` — the server (stdlib-only, localhost:8765–8774)
- `compare.html` — cross-run page (new in this variant)
- `index.html`, `detail.html`, `style.css`, `renderers.js` — single-run view, from rollout-dashboard
- `chart.umd.min.js` — vendored Chart.js 4.4.0 (MIT), so no CDN is needed
- `example/generate_runs.py` — synthetic runs for demoing without S3
