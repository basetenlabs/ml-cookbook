# autoresearch fine-tuning on Baseten

Run autonomous LoRA fine-tuning experiments by submitting training jobs to Baseten cloud GPUs.

## Setup

To set up a new experiment, work with the user to:

1. **Read settings**: Read `settings.env` for configuration (model, dataset, GPU, etc.).
2. **Agree on a run tag**: Propose a tag based on today's date (e.g., `mar18`). Update `RUN_TAG` in `settings.env`. The branch `autoresearch/<tag>` must not already exist - this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
4. **Read the in-scope files**: Read these files for full context:
   - `README.md` - repository context.
   - `training/experiment.env` - the file you modify. LoRA hyperparameters and training config.
   - `training/run.sh` - maps experiment.env variables to megatron sft flags. Don't modify.
   - `training/config.py` - Baseten training configuration. Don't modify.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. Record the baseline after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment submits a training job to Baseten. The training runs for a fixed number of iterations (`TRAIN_ITERS` in `experiment.env`). Submit it as:

```bash
truss train push training/config.py --non-interactive --remote $REMOTE --team $TEAM
```

Where `$REMOTE` and `$TEAM` come from `settings.env`. If `TEAM` is empty, omit the `--team` flag.

**What you can do (default scope):**
- Modify `training/experiment.env` - hyperparameters and training config. Everything is fair game: LoRA rank, learning rate, batch size, sequence length, recompute settings, etc.

**What you can't do (default scope):**
- Modify `training/run.sh`, `training/config.py`, or `settings.env` during the experiment loop.
- Add new files to the training directory.
- Install new packages. You can only use what's in the baseten/megatron base image.

**User-granted scope overrides**: The user's prompt can widen your scope beyond `experiment.env`. For example:
- *"You can also modify run.sh"* — unlocks changes to the training entrypoint (add flags, change the megatron sft command, add preprocessing steps, etc.).
- *"You can modify train.py"* or *"full access to training code"* — Karpathy-style: you can rewrite the training logic, change the optimizer, modify architecture, add custom loss functions, etc.
- *"You can add new files"* — lets you create helper scripts, custom datasets, or configuration files in the training directory.

When the user grants wider scope, treat those files the same way you treat `experiment.env`: commit changes, submit, measure, keep or discard. The hill-climbing loop stays the same — only the search space changes. Always prefer the minimal change that improves val_loss.

**The goal is simple: get the lowest `val_loss`.** Everything in `experiment.env` is fair game. The only constraint is that the job runs without crashing and finishes within a reasonable time.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful `val_loss` gains, but it shouldn't blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. Fewer changed parameters for the same result is a win.

**The first run**: Your very first run should always establish the baseline, so submit with `experiment.env` as-is.

## Search space guidance

When choosing what to try next, consider these dimensions:

- **LoRA rank**: 4, 8, 16, 32, 64. Higher rank = more parameters = better fit but more memory.
- **LoRA alpha**: Typically 2x the rank. Controls the scaling of LoRA updates.
- **Learning rate**: 1e-5 to 1e-3. The most impactful hyperparameter.
- **Target modules**: `all-linear` vs specific layers (e.g., `q_proj,v_proj`).
- **Batch size**: `MICRO_BATCH_SIZE` x GPUs x gradient accumulation steps = `GLOBAL_BATCH_SIZE`.
- **Recompute layers**: Trade memory for speed. More layers = less VRAM but slower.
- **Sequence length**: `MAX_LENGTH`. Longer = more context but more memory.
- **Packing**: `true` is almost always better for throughput.
- **Optimizer CPU offload**: Saves VRAM at the cost of some speed.

Start with learning rate and LoRA rank - these have the biggest impact. Then explore batch size and sequence length.

## Submit and monitor jobs

### Step 1: Submit the job

```bash
truss train push training/config.py --non-interactive --remote $REMOTE --team $TEAM
```

**CRITICAL**: Always use `--non-interactive` to prevent blocking prompts.

Parse the output for `job_id`. The push command returns it after submission.

### Step 2: Wait for the job to complete

Stream logs until the job reaches a terminal state:

```bash
truss train logs --job-id <job_id> --remote $REMOTE --non-interactive --tail
```

The `--tail` flag streams logs in real time and waits for the job to complete. The job will transition through `CREATED` -> `DEPLOYING` -> `RUNNING` -> `COMPLETED` (or `FAILED`). You can also check job status with `truss train view --job-id <job_id>`.

### Step 3: Extract results

From the logs output, look for the structured results block printed by `run.sh` at the end of training:

```
---
val_loss:         2.3456
total_seconds:    610
peak_vram_mb:     72000.0
train_iters:      100
lora_rank:        8
```

Extract `val_loss` and `peak_vram_mb` for the results log.

If the job failed, the logs will contain the error. Common failures include OOM (reduce batch size or max_length) and dataset issues.

## Log results

When an experiment finishes, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_loss	memory_gb	status	description
```

1. Git commit hash (short, 7 chars).
2. `val_loss` achieved (e.g., 2.345600) - use 0.000000 for crashes.
3. Peak memory in GB, rounded to .1f (e.g., 70.3 - divide `peak_vram_mb` by 1024) - use 0.0 for crashes.
4. Status: `keep`, `discard`, or `crash`.
5. Short text description of what this experiment tried.

Example:

```
commit	val_loss	memory_gb	status	description
a1b2c3d	2.345600	70.3	keep	baseline
b2c3d4e	2.298100	70.5	keep	increase lora_rank to 16
c3d4e5f	2.356000	70.3	discard	lower LR to 1e-5
d4e5f6g	0.000000	0.0	crash	max_length 32000 (OOM)
```

## The experiment loop

### Sequential mode (PARALLEL_JOBS=1)

LOOP (until budget exhausted or interrupted):

1. If `TOTAL_BUDGET` > 0, check experiment count. If at budget, stop and report final results.
2. Look at the git state: the current branch and commit you're on.
3. Tune `training/experiment.env` with an experimental idea by directly editing the values.
4. Git commit with a descriptive message (prefix: `exp: `).
5. Submit: `truss train push training/config.py --non-interactive --remote $REMOTE --team $TEAM`.
6. Parse `job_id` from the output.
7. Stream logs with `truss train logs --job-id <job_id> --remote $REMOTE --non-interactive --tail` until the job completes or fails.
8. Extract `val_loss` and `peak_vram_mb` from the logs.
9. If the job failed, read the logs for diagnosis. Try to fix if it's a simple bug (e.g., OOM - reduce batch size or max_length). If unfixable, log as crash and move on.
10. Record the results in `results.tsv` (don't commit `results.tsv` - leave it untracked).
11. If `val_loss` improved (lower), "advance" the branch and keep the git commit.
12. If `val_loss` is equal or worse, `git reset` back to where you started.

### Parallel mode (PARALLEL_JOBS > 1, default)

When `PARALLEL_JOBS` is set to N > 1 in `settings.env`:

LOOP (until budget exhausted or interrupted):

1. If `TOTAL_BUDGET` > 0, check remaining budget. Only generate min(N, remaining) experiments. If none remain, stop and report final results.
2. Analyze `results.tsv` history and generate experiment ideas (up to N, or remaining budget).
3. For each experiment:
   a. Save modified `experiment.env` variant to `training/variants/exp_N.env`.
   b. Copy variant to `training/experiment.env` and git commit (prefix: `exp: `).
   c. Submit: `truss train push training/config.py --non-interactive --remote $REMOTE --team $TEAM`.
   d. Record (`job_id`, `commit_hash`).
   e. Immediately `git reset --soft HEAD~1` to restore working state.
4. Stream `truss train logs --tail` for all N jobs simultaneously (use background bash jobs) until they complete.
5. Compare `val_loss` across all N experiments.
6. Cherry-pick the best improvement's commit (if any) and discard the rest.
7. Append all N results to `results.tsv`.
8. Clean up `training/variants/`.

Each `truss train push` bundles the working directory at submission time, so all N jobs capture different `experiment.env` versions even though they're submitted from the same branch.

## Key rules

- **Default: only modify `training/experiment.env`** - unless the user's prompt explicitly grants wider scope (see "User-granted scope overrides" above).
- Always use `--non-interactive` with all Truss commands. Include `--team $TEAM` if `TEAM` is set in `settings.env`.
- Log every experiment to `results.tsv` with status: `keep`, `discard`, or `crash`.
- On crash: attempt diagnosis from logs, try to fix, and if unfixable, skip and move on.

**Budget**: If `TOTAL_BUDGET` is set to a number > 0 in `settings.env`, you have a fixed number of total experiments to run. Track your experiment count and stop when you hit the budget. This lets users control GPU spend - e.g., `PARALLEL_JOBS=2` with `TOTAL_BUDGET=8` means run up to 2 concurrent jobs at a time, 8 experiments total. When `TOTAL_BUDGET=0` (default), there's no limit.

**NEVER STOP** (unless budget is reached): Once the experiment loop has begun (after the initial setup), don't pause to ask the human if you should continue. Don't ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working *indefinitely* until you're manually stopped or you exhaust your `TOTAL_BUDGET`. You're autonomous. If you run out of ideas, think harder - re-read the search space guidance, try combining previous near-misses, or try more radical parameter combinations. The loop runs until the human interrupts you or the budget is spent, period.

**Timeout**: Each experiment should take ~10-15 minutes including container startup and model download. The first job in a new project is slower because it downloads the model weights. If a job shows no progress after 20 minutes, treat it as a failure.

**Caching**: The first job in a new project is slower because it downloads model weights and the dataset. Subsequent jobs reuse the Baseten project cache (`BT_PROJECT_CACHE_DIR`), so startup is much faster.
