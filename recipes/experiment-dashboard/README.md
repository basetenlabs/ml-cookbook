# S3 Experiment Dashboard

A local, self-hosted viewer for training runs stored in S3. Every run uploads three small files (`metrics.jsonl`, `hyperparams.json`, `meta.json`) to a shared bucket prefix; this recipe syncs the prefix down, indexes it, and serves a browser dashboard for loss curves, hyperparameter filtering, and cross-run comparison. Loops and Training Jobs v1 can publish the same file contract. No vendor, no shared server, no database. S3 is the store of record, and the dashboard is a read-only viewer on `localhost` (Python stdlib only).

Forked from [`rollout-dashboard`](../loops/rollout-dashboard) — each run also gets that skill's full single-run view. If you're driving this with a Claude agent, point it at [SKILL.md](SKILL.md); this README is the human-readable setup path.

## What you see

- **Compare page (the landing page)** — a run table (run id, source, every hyperparameter, last step, final/best eval loss), filter chips for each hyperparameter that varies across runs, checkboxes to pick runs, and one overlay chart of any logged metric vs step. The metric selector lists every metric any run logged (defaults to `eval_loss`).
- **Per-run view** (click a run id) — one line chart per logged metric (`train_loss`, `eval_loss`, `learning_rate`, `grad_norm`, … whatever the run wrote), and the full hyperparameter/config table. Rollout panels (inherited from rollout-dashboard) appear when the run's `run.json` lists rollout sources — the Claude-skill flow ([SKILL.md](SKILL.md)) can populate these during detection; the plain `build_manifests.py` indexes metrics/config only.
- **Nothing is hardcoded** — every numeric key in `metrics.jsonl` becomes a chart. Scalars only: no images, no histograms.
- Dark/light theme toggle on every page; the per-run view auto-refreshes every 5 s (the compare page picks up new data on browser reload).

## Before you start — get these from your team

| You need | Get it from | Check |
| --- | --- | --- |
| S3 URI for the experiment store (`s3://<bucket>/<prefix>`) | Whoever owns your training infra | ☐ |
| AWS credentials with **read** access to that URI (`aws configure` or an `AWS_PROFILE`) | Your AWS admin | ☐ |
| AWS CLI installed (`aws --version`) | [Install docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) | ☐ |
| Python 3.9+ and a browser | Already have, most likely | ☐ |

**⛔ STOP:** do not continue until every box above is checked. If you are an agent setting this up: ask a human for the S3 URI and credentials — never guess, fabricate, or hardcode them. Verify with `aws s3 ls <S3_URI>` before proceeding. (No credentials yet? Skip to [Trying it without S3](#trying-it-without-s3).)

## What your training runs must write

One directory per run under the shared prefix:

```
s3://<bucket>/<prefix>/
  sft-llama8b-lr3e5-2026-07-28-1412/
    metrics.jsonl      # one JSON object per logged step
    hyperparams.json   # flat dict of hyperparameters, e.g. json.dump(vars(args))
    meta.json          # written once at start: {"source": "ec2", "started_at": ..., "git_sha": ...}
```

The logger is ~10 lines — call it from your training loop or an HF `TrainerCallback`:

```python
import json, time

class JsonlLogger:
    def __init__(self, path):
        self.f = open(path, "a")
    def log(self, step, **metrics):   # log(120, train_loss=1.83, learning_rate=2.4e-5, grad_norm=0.71)
        self.f.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n")
        self.f.flush()
```

Unlike W&B, system metrics (GPU utilization/memory) aren't collected automatically — log them as extra keys (`gpu_mem`, `tokens_per_sec`, …) and they chart like everything else.

Upload at run end (or periodically, for near-live curves):

```bash
aws s3 sync <run_dir> s3://<bucket>/<prefix>/<run_id>/
```

### Training Jobs v1

Training Jobs v1 injects the Baseten job ID, project ID, names, and node rank into each training container. Workspace secrets can provide customer-managed S3 credentials. The included adapter uses those values to produce and upload the three dashboard files from the primary training process.

See [`training_jobs_v1/README.md`](training_jobs_v1/README.md) for the `config.py` secret mapping and the training-loop integration. The adapter uses `BT_TRAINING_JOB_ID` as the run ID and sets `source` to `baseten-training-jobs-v1`.

Use a customer-managed experiment bucket for this workflow. Do not place dashboard files in the Baseten checkpoint directory. Baseten manages that directory as model output, while this dashboard needs a shared S3 prefix that the team can read directly.

## Setup and run

From this directory (or a copy of it anywhere):

```bash
S3_URI=s3://<bucket>/<prefix> ./sync.sh     # pull the store down into ./runs/
S3_URI=s3://<bucket>/<prefix> python3 build_manifests.py   # index: per-run run.json + experiments.json
python3 dashboard.py                        # serve on http://localhost:8765
```

Open http://localhost:8765 — the compare page lists all runs with filter chips and an overlay chart of any logged metric; each run id links to its full single-run dashboard. To pull new runs later, re-run the first two commands (the server picks changes up on the next refresh, no restart needed).

## Trying it without S3

```bash
python3 example/generate_runs.py   # writes 5 synthetic runs into ./runs/ (seeded, reproducible)
python3 build_manifests.py
python3 dashboard.py
```

## Troubleshooting

- **`AccessDenied` / `Unable to locate credentials` from sync.sh** — your AWS setup isn't done; go back to "Before you start".
- **Port in use** — the server tries 8765–8774 automatically; check the startup line for the port it picked.
- **Empty charts / missing metric in the selector** — charts are built from the numeric keys in `metrics.jsonl` (everything except `step`/`time`); check what your logger writes, then re-run `build_manifests.py`.
- **Stale data** — re-run `./sync.sh` and `build_manifests.py`; then refresh the browser.
- **Locked-down browser / no CDN** — already handled: Chart.js is vendored as `chart.umd.min.js`.
