# Loops Cookbook

Starter recipes for training models with [Loops](https://pypi.org/project/baseten-loops/), Baseten's SDK for online training. Provision a trainer, run forward-backward passes, take optimizer steps, sync weights to a paired sampler, and generate completions — all from a Python script, no infrastructure management required.

These recipes use [tinker-cookbook](https://pypi.org/project/tinker-cookbook/)'s training loops and environment abstractions, backed by Loops via the [`baseten-loops-tinker`](https://pypi.org/project/baseten-loops-tinker/) compatibility shim. The shim provides the `tinker` package namespace on top of the Loops SDK, so tinker-cookbook code runs against Baseten without modification — `pyproject.toml` uses a uv [dependency override](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides) to keep the original `tinker` package from being installed.

## Recipes

| Recipe | What it shows |
| --- | --- |
| [`sft/train_sft.py`](sft/train_sft.py) | Supervised fine-tuning on the [no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) chat dataset. Swap in your own JSONL conversations to fine-tune on your data. |
| [`rl/train_grpo.py`](rl/train_grpo.py) | GRPO on GSM8K math problems — synchronous sample-then-train loop. |
| [`rl/train_grpo_async.py`](rl/train_grpo_async.py) | Async GRPO with bounded off-policy sampling — rollouts and optimizer steps run concurrently. |
| [`multiturn_rl/train_twenty_questions.py`](multiturn_rl/train_twenty_questions.py) | Multi-turn RL: the policy plays twenty questions against a frozen answerer model served by a second sampler. [`env.py`](multiturn_rl/env.py) is the template for building your own multi-turn environment. |

## Setup

You need [uv](https://docs.astral.sh/uv/), a [Baseten account](https://app.baseten.co/signup), and a training project. Then:

```bash
export BASETEN_API_KEY="your-api-key"
export LOOPS_PROJECT_ID="proj_abc123"
uv sync
```

## Run

```bash
uv run sft/train_sft.py
uv run rl/train_grpo.py
uv run rl/train_grpo_async.py
uv run multiturn_rl/train_twenty_questions.py
```

The first run provisions a trainer (and a paired sampler for the RL recipes) in your training project, which can take a few minutes. Subsequent runs reuse the live servers.

Every config field can be overridden from the command line:

```bash
uv run rl/train_grpo.py model_name=Qwen/Qwen3.5-27B learning_rate=1e-5 max_tokens=512
```

Training metrics land in the recipe's `log_path` (under `/tmp/loops-cookbook/` by default) as `metrics.jsonl`; pass `wandb_project=...` to log to Weights & Biases.

## Async RL and off-policy bounds

In `rl/train_grpo_async.py`, rollout workers generate trajectory groups continuously while the training loop consumes them — sampling never waits for the optimizer and vice versa. The cost is staleness: a rollout may have been sampled from a policy several optimizer steps old.

`max_steps_off_policy` bounds that staleness. Each trajectory group is tagged with the policy version it was sampled from; groups more than `max_steps_off_policy` steps behind the current trainer step are requeued instead of trained on. This rides on Loops' weight-versioning semantics:

- Every `optim_step` on the trainer bumps its **policy version**.
- After each step, the updated LoRA adapter is published to the paired sampler, which hot-swaps it in.
- Sampling requests carry an `X-Min-Policy-Version` floor — the sampler blocks until its live adapter reaches that version, so rollouts are never served by weights older than the version they were pinned to, and every sample result reports the policy version that produced it.

`max_steps_off_policy=2` is a reasonable default; raise it for more pipelining throughput, lower it to stay closer to on-policy.

## Choosing a base model

Defaults here use Qwen3.5 models, which are supported by Loops trainers and samplers. Pick the smallest model that works for your task to keep iteration fast — `Qwen/Qwen3.5-4B` is a good starting point, and `Qwen/Qwen3-0.6B` trains in minutes if you just want to validate an environment before scaling up.

## Visualizing your rollouts

Use the `rollouts-dashboard` skill to visualize your rollouts as your training progresses. This skill helps your agents build helpful data visualizations for RL use cases.

## License and attribution

The recipes in this directory are adapted from [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), Copyright 2025 Thinking Machines Lab, licensed under the Apache License 2.0 (see [LICENSE](LICENSE) in this directory). Each adapted file carries an attribution header noting it was modified by Baseten. This Apache-2.0 licensing applies to the contents of `recipes/loops/`; the rest of this repository is MIT-licensed.
