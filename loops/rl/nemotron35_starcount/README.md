# Nemotron 3.5 visual-counting LoRA GRPO

Fine-tune a LoRA adapter by generating answers to synthetic star-count images and
rewarding the correct count. This recipe uses **Baseten Loops SDK 0.28.0** in its
own pinned Python environment, independently of the cookbook's parent environment.
Images are generated locally; model training and sampling run on managed GPUs.

## Recipe

| Setting | Value |
| --- | --- |
| Managed model | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` |
| Execution | Dedicated Loops trainer and paired sampler; GPU layout managed |
| Dataset | 1,024 training and 256 separate validation images; fixed seeds, bytes and order |
| Updates | 50, starting fresh by default |
| Rollouts per update | 16 prompts, eight responses each |
| LoRA | Rank 64; remaining adapter settings are managed |
| Context / generation | 4,096 / 256 tokens; overlong input is rejected |
| Sampling | Temperature 1 and top-p 1 for training; greedy validation |
| Learning rate | 3e-6, with a ten-update warmup from 3e-8, then constant |
| Optimizer | Adam, betas 0.9/0.99, epsilon 1e-8, weight decay 0, clip norm 1 |
| Validation | All 256 rows before training and every two updates |
| Checkpoints | Trainer state at updates 10, 20, 30, 40 and 50 |

The reward is binary: the first literal `\boxed{integer}` in the final answer
must equal the image metadata's target-color count. Generation starts inside
the thinking prefix; answers without the closing `</think>` separator receive
zero reward. Boxes in reasoning are ignored. Training uses leave-one-out group
advantages and the pinned clipped policy loss implemented in `core.py`/`loss.py`.

This is **GRPO-style group-relative RL**: the baseline and standard deviation
use the other seven responses, rather than canonical whole-group normalization.
Policy ratios are clipped to 0.8–1.28, detached sampler/trainer importance weights
to 0.5–2.0, and loss reduction is sequence-mean. There is no critic or reference
KL penalty. Negative-advantage responses remain in training. Reaching the
256-token generation cap alone does not exclude a response; only the total
context limit triggers overlong filtering.

## Requirements

- Python 3.13.14 and [uv](https://docs.astral.sh/uv/).
- A Baseten API key for a workspace with this model enabled for vision RL and
  enough trainer and sampler capacity. Availability and placement depend on your
  workspace; this recipe does not select a GPU type or parallelism layout.
- A Weights & Biases API key and access to the project you select.
- Access to the public Hugging Face model repository for three small metadata
  files. The exact public commit and file hashes are pinned in
  `data_manifest.json`; no manually supplied tokenizer folder is needed.
  No model weights, generated data, outputs or credentials are distributed.

**Training allocates billable remote GPUs for both a trainer and its paired
sampler.** Keep the Python client running and connected on a stable machine.
Closing the client, sleeping a laptop or forcibly killing Python is not proof
that billing has stopped. Setup, generation, tests and offline preflight do not
allocate GPUs.

## 1. Install and generate the frozen data

From the cookbook checkout:

```bash
cd loops/rl/nemotron35_starcount
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
uv sync --locked
uv run --locked python generate_data.py
```

The generator runs offline without credentials. Both complete JSONL files must
match these SHA256s before they are published in ignored `data/`:

| Split | Rows | SHA256 |
| --- | --- | --- |
| Train | 1,024 | `6b63bf98dd2107a35b6ed442dbb5a8003ba15c8053b66dc37f9fbe617d24b307` |
| Validation | 256 | `76a8a383ab459826117bbf11699c5908b10d539d9fd21b437d0c1e9149f096b8` |

Repeating generation verifies correct cached files. Mismatching existing data
is not overwritten; investigate it or use a fresh recipe copy.

## 2. Prepare public tokenizer metadata automatically

```bash
uv run --locked python prepare_assets.py --record-source
```

This downloads the manifest-pinned public snapshot directly from
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`. No HF login, manual
tokenizer directory or user-supplied commit is required. Source provenance is
recorded in ignored `tokenizer_source.json`; a matching cache is reused offline.
For an optional offline import, use `prepare_assets.py --tokenizer-dir
/path/to/matching/tokenizer`. Any override must match the same pinned hashes.

Only `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` are
imported into ignored `tokenizer/`. Their three hashes must match
`data_manifest.json`; model weights and remote Python code are not downloaded.
The small `processor_config.json` is included. A missing file, access denial or
hash mismatch blocks setup before any GPU allocation. Use a fresh recipe copy
to investigate different metadata; do not replace the manifest to bypass checks.

## 3. Audit locally and configure your accounts

```bash
uv run --locked python preflight.py --offline
uv run --locked python -m unittest discover -s tests -v

export BASETEN_API_KEY="your-baseten-api-key"
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_PROJECT="loops-starcount"
```

Set `BASETEN_TEAM` and `WANDB_ENTITY` when your accounts require them. Otherwise
the Baseten default team and your W&B defaults apply. Train, preflight and
cleanup also accept `--env-file /path/to/keys`; files are read only when
explicitly supplied, and existing shell values take precedence. Never commit
that file. Unset `LOOPS_REUSE_FROM_RUN_ID`, `LOOPS_REUSE_FROM_SESSION_ID` and
`INTERNAL_LOOPS_BACKEND_VERSION` before training.

An optional online capability check reads model access without provisioning:

```bash
uv run --locked python preflight.py
```

Preflight verifies every row, image, expanded prompt, tokenizer hash and complete
prompt-plus-generation token bound. Resizing uses the bundled geometry rules;
float resized pixels are quantized to PNG for transport. Identity with the
managed server's processor is not established by this local audit.

## 4. Run a three-update smoke test

This command allocates a fresh trainer and paired sampler, evaluates all 256
baseline examples, performs three updates, validates and saves checkpoint 3:

```bash
STAMP=$(date -u +%Y%m%d_%H%M%S)
SMOKE="baseten_starcount_smoke_${STAMP}"
uv run --locked python train.py --steps 3 --name "$SMOKE"
uv run --locked python cleanup.py --output "outputs/$SMOKE" --yes
```

Check the completion, checkpoint registry and cleanup reports before a full
run. The smoke verifies wiring, not improved accuracy, and does not
automatically start another run.

## 5. Run all 50 updates

Choose a new name for each invocation:

```bash
STAMP=$(date -u +%Y%m%d_%H%M%S)
RUN="baseten_starcount_full_${STAMP}"
uv run --locked python train.py --name "$RUN"
```

This starts the full 50-update run directly. It creates a new owned trainer and
paired sampler, evaluates the baseline, performs the updates and saves results
under `outputs/$RUN`. Existing output directories are not overwritten. W&B
records validation accuracy, policy loss, learning rate and rollout artifacts.

Check `completion.json` for `complete: true` and 50 updates, validation coverage
of all 256 rows, and `checkpoints.json`/`registered_checkpoints.json` for the
saved states. Check cleanup separately as described below; a training completion
record is not a resource shutdown record.

To resume an interrupted **`cookbook-v2`** run with a matching recipe/data
fingerprint, supply its `resume.json` and a new name:

```bash
uv run --locked python train.py \
  --resume "outputs/PREVIOUS_RUN/resume.json" \
  --name "baseten_starcount_resume_${STAMP}"
```

Resume restores weights and optimizer into a new owned run. Outputs from the
older recipe versions are incompatible with this contribution's
fingerprint; preserve them and start a fresh run.

## Cleanup and results

The normal exit path attempts to deactivate only this invocation's trainer and
paired sampler. `--keep-active` deliberately leaves them billable. After an
interruption, failed cleanup, or for an explicit final verification, use:

```bash
uv run --locked python cleanup.py --output "outputs/$RUN" --yes
```

With a key file, add `--env-file /path/to/keys`. Inspect `deactivation.json`:
`requested: true` is distinct from `verified: true`. A failed trainer is recorded
as provider-terminal, not as a measured GPU-replica count. A paired sampler
requires `INACTIVE` status and zero active replicas; an inactive run alone is
insufficient. If ownership is ambiguous or the sharing check is unavailable,
cleanup stops with an unverified report and exact manual commands rather than
stopping another user's resources. If a forced kill happened before IDs were recorded,
identify this uniquely named run in the dashboard before taking action. Do not
stop unrelated resources.

## Observed accuracy

A completed 50-update Baseten Loops run using the BF16 model and LoRA rank 64
improved accuracy on the same 256-example validation set:

| Evaluation | Correct answers | Accuracy |
| --- | --- | --- |
| Before training | 30 / 256 | 11.72% |
| After 50 updates | 145 / 256 | 56.64% |

This is an improvement of **44.92 percentage points**. The
[completed W&B run](https://wandb.ai/nvidia/Super-3.5-benchmark/runs/xx00rtyj)
contains the metrics and rollout evidence. These are historical observations
from the earlier `standalone-v1` recipe, not guaranteed outcomes for the current
`cookbook-v2` recipe. The current recipe has passed a three-update functional
smoke test; its full 50-update accuracy has not yet been verified.

Validation is monitored during training, not held back as a final test set.
The reward requires a final boxed answer within 256 generated tokens, so
accuracy reflects counting plus output-format and token-budget compliance.
This is a synthetic task, not a general real-world object-counting benchmark.
W&B logs metrics and rollout
tables with `baseten_starcount_` run names.

See [Baseten Loops](https://docs.baseten.co/loops/overview) and its
[supported models](https://docs.baseten.co/loops/supported-models).
Licensing and retained attribution are in `LICENSE` and `NOTICE`.
