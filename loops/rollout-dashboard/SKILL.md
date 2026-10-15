---
name: rollout-dashboard
description: Build a local browser dashboard for digging into AI training rollouts — the text the model actually produced during training. Works on any training output (RL, SFT, DPO, GRPO, eval suites, custom research code, Tinker, prime-rl, verifiers, HuggingFace, anything else) by detecting where the rollouts live and rendering them as a browsable, click-to-read interface. Use this skill whenever the user wants to inspect a training run, debug model outputs, browse generations, see what the model wrote, or "dig into rollouts" — even if they don't say "dashboard". Rollouts are the focus; metrics and configs are secondary context.
---

# Training Explorer

Build a browser dashboard that puts **rollouts first**. Researchers come to a training run to read what the model wrote. Everything else (metrics, configs, hyperparameters) is supporting context.

This skill is two things:
1. **Detection logic** (this file) — finds rollouts in the run directory, identifies the relevant fields, writes a small JSON config.
2. **A pre-built renderer** (`renderer/` folder) — actually renders the dashboard. Already written, already styled, already tested. Don't regenerate it; copy it.

The detection writes `run.json` describing what was found. The renderer reads `run.json` and renders. They're decoupled.

## What you do, in order

1. Use the Read/Glob/Grep tools to walk the run directory at depth 3.
2. Score every structured file for rollout-likelihood (table below).
3. For each file scoring ≥3, identify the **five rollout fields**: id, content, score, step, group.
4. Detect metrics files and config files (the "context" region).
5. Write `<run_dir>/run.json` describing what was found.
6. Copy the `renderer/` folder contents into `<run_dir>/`.
7. Tell the user to run `python <run_dir>/dashboard.py` and stop.

**Do not** edit the renderer. **Do not** regenerate `dashboard.py`. **Do not** "improve" the templates. The renderer is a closed system that reads `run.json`. Your job is to produce a good `run.json`.

## Operating rules

- Use Read/Glob/Grep — never `python -c` for inspection.
- Peek files; don't slurp. Use `head -3` equivalent.
- Don't run the dashboard. Don't curl it. Hand it off and stop.
- If a run directory was provided as an argument, use it directly. Only ask if none was given.

## Phase 1: Find rollout sources

Walk the directory. For each structured file (`*.jsonl`, `*.json`, `*.csv`, `*.html`), compute a **rollout-likelihood score**:

| Signal | Score |
|---|---|
| Has a field with long strings (≥200 chars avg across sample) | +1 |
| Those long strings contain turn markers matching `Turn N - Speaker: ...` | +3 |
| Records look like `{prompt, completion}` pairs (both long strings) | +3 |
| Records look like `{chosen, rejected}` pairs (both long strings) | +3 |
| Has a field that is a list of `{role, content}`-shaped dicts | +3 |
| Has a dict field shaped like HTML AST: has `tag` + `children` keys | +2 |
| File is in a directory named `rollouts/`, `samples/`, `outputs/`, `trajectories/`, `generations/` | +1 |
| File path matches `iteration_*/`, `step_*/`, `epoch_*/` pattern | +1 |
| File is a typed event stream (has `type` field with values like `rollout_*`, `assistant_turn`, `tool_call`) | +3 |
| File is `.html` and contains `<` chars with classes like `lt-`, `rollout-`, `turn-` | +2 |
| Filename ends with `_logtree.json` or has a `root` field containing nodes with `data.type == "conversation"` | +4 |

**Score ≥ 3 → this file is a rollout source.** Lower scores → context or ignored.

### Prefer structured over rendered

When the same content exists in two forms — a `*.html` rendered view and a `*_logtree.json` structured export side-by-side in the same directory — **always pick the JSON**. It's the same data but has properly structured conversation nodes instead of flattened HTML where parallel trajectories interleave. List the HTML under `ignored` with reason "rendered view of <json file>".

If only `.html` exists (older Tinker runs, or other frameworks that don't dump JSON), use it — the renderer's HTML parser will produce an AST. But expect parallel-trajectory cases to render less cleanly.

### Inspection procedure for each file

To compute these signals, sample the first 3-5 records (for JSONL: `head -n 5 file.jsonl`; for JSON: read the whole thing if small, head if large; for CSV: read the header + first 3 rows). For each sampled record, walk its keys and value shapes.

A "field with long strings" means: at least one key whose values are strings averaging ≥200 chars. Compute the average across sampled records.

For HTML AST detection: check if any field is a dict where the top-level keys include `tag` and `children` (and usually `attrs`). One sample is enough.

For the conversation list signal: check if any field is a list of dicts where each dict has at least one key from `{role, speaker, from, author}` AND at least one key from `{content, text, message, value}`. Sample the first 2-3 list items.

### Counting rollouts in logtree files

A single logtree JSON/HTML file usually contains **many parallel rollouts** (e.g., 8 trajectories in a GRPO group, ×4 groups logged = 32 rollouts per iteration). Don't count files; count *conversation nodes* inside the AST.

Conceptually, walk the `root` tree looking for nodes with `data.type == "conversation"` (Tinker format) or any node with a `messages` field that's a list of `{role, content}`-shaped dicts. Each one is a rollout.

For the skill, you can approximate this by reading one file from the glob and counting matches. Multiply by the number of files matching the glob. Or, peek at one file with grep: `grep -c '"type":\s*"conversation"' iteration_*/train_logtree.json | head -1` gives a per-file count.

The renderer (`dashboard.py`) does the actual splitting at request time, so the `n_rollouts` field in `run.json` is informational only — it's used for the dashboard counts. It's better to under-estimate than over-promise; the actual rollout count will be reported correctly once the dashboard loads.

### Multi-file rollout sources

If multiple files share a glob pattern (e.g., `iteration_*/rollouts.jsonl` × N, `samples/step_*.jsonl` × N, `iteration_*/train.html` × N), they likely form one logical source. Merge them: one entry in `run.json` with a `glob` field, total record count summed across files. The renderer treats them as a single browsable source.

The merge rule: if two or more files at the same depth in the directory tree share the same name (after replacing digits with `*`), they merge.

## Phase 2: Identify the five rollout fields

For each rollout source, peek at a few records and identify these fields. Only `content` is required.

### `id` — unique per rollout

Look for a string or int field with one value per record. Common names: `rollout_id`, `id`, `uuid`, `trajectory_id`, `sample_id`. If multiple are unique, prefer shorter names. If none are unique, set to `null` — the renderer will synthesize `row-0`, `row-1`, etc.

**For `logtree_ast` sources, set `id` to `null`** unless the AST nodes themselves have an obvious id field. The renderer splits the logtree into N conversation records and synthesizes ids per record.

For event streams: `id` is `rollout_id` (or whatever field groups events into rollouts). Detect this by finding the string field that has many distinct values, where each value appears in multiple consecutive events.

### `content` — the actual rollout text

This is the field that scored the rollout-source points above. Possibilities, in priority order:

1. **Field of type `list`** matching the conversation_list shape (`[{role, content}, ...]`) → `format: "conversation_list"`.
2. **Field of type `string`** with turn markers (`Turn N - X: ...`) → `format: "turn_text"`.
3. **Field of type `dict`** with HTML AST shape (`{tag, children, attrs}`) → `format: "logtree_ast"`.
4. **Two fields, prompt-like + completion-like** (string types, both long, names from `PROMPT_KEYS = {prompt, input, question, instruction, query}` and `COMPLETION_KEYS = {completion, output, answer, response, target}`) → `format: "prompt_completion"`, write both field names.
5. **The whole file is a typed event stream** (records have `type` field with rollout-event-like values) → `format: "event_stream"`, content is the list of events keyed by `id`.
6. **Long string field, but none of the above** → `format: "raw_text"`, render as preformatted text.

Pick the highest-priority match. Don't guess at multiple formats per source.

### `score` — quality signal

A float field with cardinality > 1, ideally in [0, 1]. Common names but **not required to match**: `reward`, `score`, `total_reward`, `final_reward`, `quality`, `accuracy`. Pick the first float field with `0 <= min <= max <= 1` and cardinality > 1. If none qualify, try any float field with `cardinality > 1`. If still none, `null`.

Exclude: fields that are monotonically increasing (those are indices/progress, not scores).

**For `logtree_ast` sources**, the renderer splits the file into per-trajectory records using one of two paths depending on the logtree format:

- **Tinker JSON format** (`data.type == "conversation"` nodes): metadata inside the `data` field is surfaced as `meta_<key>` on each record. Use `score: "meta_reward"`, `group: "meta_group_idx"`. Peek inside the JSON to see what metadata keys are present.
- **lt-* HTML library format** (parallel trajectories as interleaved `lt-p` paragraphs, no explicit conversation nodes in JSON): the server directly sets `won`, `group_idx`, `traj_idx`, and `secret` on each split record. Use `score: "won"`, `group: "group_idx"`. You can detect this format by checking if the logtree JSON has a `root` field whose children have `attrs.class` values like `lt-section`, `lt-h2`, `lt-p` rather than `data.type == "conversation"` nodes.

### `step` — training step

An integer field that's monotonically increasing across records, or matches the directory pattern (e.g., parent dir is `iteration_NNNNNN/`). Common names: `step`, `iteration`, `epoch`, `global_step`. If extracted from directory name, store as `step_from_path: true` in the config so the renderer knows.

### `group` — bucket / category

A low-cardinality string field where the cardinality is between 2 and (n_records / 2) — so each group has at least 2 items on average. Common names: `task`, `prompt_id`, `group_id`, `split`. If none, `null` and the renderer shows a flat list instead of a group grid.

## Phase 3: Identify context (metrics + config)

For files that scored < 3 on rollout-likelihood:

### Metrics

A file is metrics if:
- It's JSONL/JSON/CSV
- It has at least one monotonically-increasing integer field (step-like)
- AND at least 2 other numeric fields

Examples: `metrics.jsonl`, `trainer_state.json` (use `log_history` array), `*.csv` with a `step`/`iteration`/`epoch` column.

Identify the **x-axis field** (first monotonic integer found) and list all numeric fields with cardinality > 1 as the plottable series.

For `trainer_state.json` specifically: the records live under `log_history`. Use those. (One concession to a well-known format.)

### Config

Anything that looks like hyperparameters: a JSON/YAML file with mostly scalar values, in the run root or in a `config/` subdir. Common names: `config.json`, `config.yaml`, `args.json`, `hparams.json`, `run_config.json`.

If the metrics-source detector found a file but it's actually a config (e.g., trainer_state.json has training config alongside log_history), use both: config from the top-level keys, metrics from `log_history`.

### Everything else

Anything that's neither rollout source, metrics, nor config — list under `ignored` with a one-line reason. Things to silently ignore: `*.bin`, `*.safetensors`, `*.pt`, `*.ckpt`, `*.pkl`, `*.npy`, `*.npz`, `wandb/`, `.git/`, `__pycache__/`, `tfevents.*`.

For `tfevents.*` specifically, mention it in the startup message — researchers can use `tensorboard --logdir <dir>` for those.

## Phase 4: Write `run.json`

Schema:

```json
{
  "run_id": "directory_name",
  "started_at": "ISO 8601, optional (use file mtime)",
  "last_updated": "ISO 8601 (latest mtime)",

  "rollout_sources": [
    {
      "name": "human-readable name, derived from filename",
      "file": "relative/path.jsonl",
      "glob": "relative/iteration_*/rollouts.jsonl (only if multi-file)",
      "n_rollouts": 4049,
      "format": "conversation_list | turn_text | logtree_ast | prompt_completion | event_stream | raw_text",
      "fields": {
        "id": "rollout_id",
        "content": "events",
        "content_prompt": "prompt (only for prompt_completion)",
        "content_completion": "completion (only for prompt_completion)",
        "score": "total_reward",
        "step": "step",
        "step_from_path": false,
        "group": "task"
      }
    }
  ],

  "metrics": {
    "file": "metrics.jsonl",
    "x_axis": "step",
    "series": ["reward", "loss", "lr", "..."]
  },

  "config": {
    "file": "config.yaml",
    "values": { "learning_rate": 3e-05, "model_name": "..." }
  },

  "ignored": [
    {"path": "checkpoint_*.pt", "reason": "binary"},
    {"path": "wandb/", "reason": "external tracker"}
  ],

  "tfevents": ["events.out.tfevents.*"]
}
```

Notes:
- `metrics` is a single object, not a list. If multiple metrics files exist, pick the most comprehensive one and list the others under `ignored`.
- `config` `values` should be the flat key-value pairs; nested structures get JSON-stringified.
- Paths are relative to the run directory.
- All keys except `run_id` and `rollout_sources` are optional. The renderer handles missing keys gracefully.
- `rollout_sources` can be empty if no rollouts were found — the dashboard degrades to metrics-and-config view.

### Example: Tinker run (lt-* HTML library format)

This is the most common Tinker format. The logtree JSON files contain interleaved `lt-p` paragraphs (no explicit `data.type == "conversation"` nodes). The server extracts per-trajectory records and sets `won`, `group_idx`, `traj_idx` directly.

```json
{
  "run_id": "Qwen3-4B-Instruct-2507-8group-64batch-3e-05lr-2026-05-19-12-06",
  "rollout_sources": [
    {
      "name": "Training rollouts",
      "file": "iteration_000000/train_logtree.json",
      "glob": "iteration_*/train_logtree.json",
      "n_rollouts": 608,
      "notes": "19 iterations × 32 trajectories (4 groups × 8 each). lt-* logtree format — server splits into per-trajectory records with won/group_idx/traj_idx fields.",
      "format": "logtree_ast",
      "fields": {
        "id": null,
        "content": "root",
        "step": "iteration",
        "step_from_path": true,
        "score": "won",
        "group": "group_idx"
      }
    },
    {
      "name": "Eval rollouts",
      "file": "iteration_000000/eval_test_logtree.json",
      "glob": "iteration_*/eval_test_logtree.json",
      "n_rollouts": 512,
      "format": "logtree_ast",
      "fields": {
        "id": null,
        "content": "root",
        "step": "iteration",
        "step_from_path": true,
        "score": "won",
        "group": "group_idx"
      }
    }
  ],
  "metrics": {
    "file": "metrics.jsonl",
    "x_axis": "step",
    "series": ["env/all/reward/total", "test/env/all/reward/total", "optim/entropy", "..."]
  },
  "config": {
    "file": "config.json",
    "values": {"learning_rate": 3e-05, "model_name": "Qwen/Qwen3-4B-Instruct-2507", "...": "..."}
  },
  "ignored": [
    {"path": "iteration_*/train.html", "reason": "rendered view of train_logtree.json"},
    {"path": "iteration_*/eval_test.html", "reason": "rendered view of eval_test_logtree.json"},
    {"path": "iteration_*/train_rollout_summaries.jsonl", "reason": "per-trajectory metrics only; conversation content lives in logtree files"},
    {"path": "iteration_*/eval_test_rollout_summaries.jsonl", "reason": "per-trajectory metrics only; conversation content lives in logtree files"}
  ]
}
```

### Example: Tinker run (JSON conversation format)

Older or differently-configured Tinker runs export JSON logtrees with explicit `data.type == "conversation"` nodes. Metadata inside `data` is surfaced as `meta_<key>` on each split record.

```json
{
  "rollout_sources": [{
    "name": "Training rollouts",
    "glob": "iteration_*/train_logtree.json",
    "format": "logtree_ast",
    "fields": {
      "id": null,
      "content": "root",
      "step": "iteration",
      "step_from_path": true,
      "score": "meta_reward",
      "group": "meta_group_idx"
    }
  }]
}
```

**Important:** The rollout_summaries.jsonl files don't contain rollout content (only metadata). Always drop them as rollout sources — the logtree files have the actual conversations and carry per-trajectory metadata themselves.

### Example: prime-rl run

```json
{
  "run_id": "run_q36_27b_128k_b256_cp2",
  "rollout_sources": [
    {
      "name": "Rollouts",
      "file": "events/orchestrator.jsonl",
      "n_rollouts": 4049,
      "format": "event_stream",
      "fields": {"id": "rollout_id", "content": "events", "score": "reward",
                 "step": "step", "group": "task"}
    }
  ],
  "metrics": {"file": "logs/orchestrator.log", "x_axis": "step",
              "series": ["reward", "n_trained", "seq_length"]}
}
```

### Example: SFT run

```json
{
  "run_id": "sft-llama-3-8b",
  "rollout_sources": [
    {
      "name": "Samples",
      "file": "samples/step_*.jsonl",
      "glob": "samples/step_*.jsonl",
      "n_rollouts": 1200,
      "format": "prompt_completion",
      "fields": {"id": "sample_id", "content_prompt": "prompt",
                 "content_completion": "completion",
                 "score": "reward", "step": "step"}
    }
  ],
  "metrics": {"file": "trainer_state.json", "x_axis": "step",
              "series": ["loss", "eval_loss", "learning_rate"]}
}
```

## Phase 5: Copy the renderer

After writing `run.json`, copy these five files from the skill directory (same folder as this `SKILL.md`) into the run directory:

- `dashboard.py` → `<run_dir>/dashboard.py`
- `index.html`   → `<run_dir>/index.html`
- `detail.html`  → `<run_dir>/detail.html`
- `style.css`    → `<run_dir>/style.css`
- `renderers.js` → `<run_dir>/renderers.js`

The renderer is stdlib-only and reads files at request time. No build step.

## Phase 6: Hand off

Tell the user:

```
Dashboard ready at <run_dir>/.

To start:
  cd <run_dir> && python dashboard.py

Then open http://localhost:8765 in your browser.

Detected:
  <N> rollout sources (<source names>)
  metrics: <file or "none">
  config: <file or "none">
  ignored: <count>

If you want to tweak the look, see CUSTOMIZATION.md in the run directory.
```

Don't run the dashboard yourself. Don't curl it. Stop.

## When detection finds nothing

If no rollout sources scored ≥3 and no metrics file was found, print:

```
No rollouts or metrics detected in <run_dir>.

Found these files but didn't recognize them:
  <list>

If one of these is your rollout file, edit run.json manually and add it
to rollout_sources, then start the dashboard.
```

Write a minimal `run.json` with empty arrays and the unrecognized files listed. The renderer will show the schema + ignored list so the user can edit it.

## The hard part

The hard part of this skill is **identifying the five fields correctly per source**. Get this wrong and the dashboard renders nothing useful, even with a perfect renderer.

Heuristics that help:
- When in doubt about which string field is content vs metadata, pick the longer one (by average length).
- When in doubt about which integer field is step vs index, prefer monotonic + matches path patterns.
- When in doubt about score, prefer floats already in [0,1] over those that need scaling.
- When a field could plausibly be id OR group, prefer id if cardinality == n_records, group otherwise.
- For event streams, the rollout boundary is usually `rollout_start` and `rollout_end` events; everything between with the same `rollout_id` is one rollout's content.

When you can't decide, write what seems most reasonable and add a note in `run.json` under a `notes` key — the renderer ignores it but the user can see it if they inspect the file:

```json
"rollout_sources": [{
  "name": "Samples",
  "notes": "Picked 'output' as content over 'prompt' because it was longer on average. Adjust if wrong.",
  ...
}]
```

## Reference

- `dashboard.py` — the server
- `index.html` — main page template
- `detail.html` — side-panel template
- `style.css` — all styling, CSS vars at the top for easy edits
- `renderers.js` — format renderers (one function per format: `conversation_list`, `turn_text`, `logtree_ast`, `prompt_completion`, `event_stream`, `raw_text`, `trajectory`, `group_turns`)
- `CUSTOMIZATION.md` — map of "if user asks for X, edit Y"
