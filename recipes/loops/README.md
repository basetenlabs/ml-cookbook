# Loops Cookbook

Starter recipes for training models with [Loops](https://pypi.org/project/baseten-loops/), Baseten's SDK for online training. Provision a trainer, run forward-backward passes, take optimizer steps, sync weights to a paired sampler, and generate completions — all from a Python script, no infrastructure management required.

These recipes use [tinker-cookbook](https://pypi.org/project/tinker-cookbook/)'s training loops and environment abstractions, backed by Loops via the [`baseten-loops-tinker`](https://pypi.org/project/baseten-loops-tinker/) compatibility shim. The shim provides the `tinker` package namespace on top of the Loops SDK, so tinker-cookbook code runs against Baseten without modification — `pyproject.toml` uses a uv [dependency override](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides) to keep the original `tinker` package from being installed.

## Recipes

| Recipe | What it shows |
| --- | --- |
| [`sft/train_sft.py`](sft/train_sft.py) | Supervised fine-tuning on the [no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) chat dataset. Swap in your own JSONL conversations to fine-tune on your data. |
| [`sft/train_sft_glm.py`](sft/train_sft_glm.py) | SFT for [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) — a Loops-supported model tinker-cookbook doesn't know. Shows the bring-your-own-model pattern: a custom renderer matching GLM's chat template (verified token-for-token against `apply_chat_template` before any GPU spend), registered via tinker-cookbook's renderer/tokenizer registries. |
| [`rl/train_grpo.py`](rl/train_grpo.py) | GRPO on GSM8K math problems — synchronous sample-then-train loop. |
| [`rl/train_grpo_async.py`](rl/train_grpo_async.py) | Async GRPO with bounded off-policy sampling — rollouts and optimizer steps run concurrently. |
| [`rl/train_math_async.py`](rl/train_math_async.py) | Async RL on [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K) — combines bounded off-policy training with warm-start sampler provisioning, so long variable-length solutions never stall the trainer. |
| [`multiturn_rl/train_twenty_questions.py`](multiturn_rl/train_twenty_questions.py) | Multi-turn RL: the policy plays twenty questions against a frozen answerer model served by a second sampler. [`env.py`](multiturn_rl/env.py) is the template for building your own multi-turn environment. |
| [`multiturn_rl/train_twenty_questions_async.py`](multiturn_rl/train_twenty_questions_async.py) | Async multi-turn RL: the same twenty-questions environment with rollout workers playing games continuously while the trainer takes optimizer steps. |

## Setup

You need [uv](https://docs.astral.sh/uv/) and a [Baseten account](https://app.baseten.co/signup). Then:

```bash
export BASETEN_API_KEY="your-api-key"
uv sync
```

## Run

```bash
uv run sft/train_sft.py
uv run rl/train_grpo.py
uv run rl/train_grpo_async.py
uv run rl/train_math_async.py
uv run multiturn_rl/train_twenty_questions.py
uv run multiturn_rl/train_twenty_questions_async.py
```

The first run provisions a trainer (and a paired sampler for the RL recipes) in your training project, which can take a few minutes. Subsequent runs reuse the live servers.

Every config field can be overridden from the command line:

```bash
uv run rl/train_grpo.py model_name=Qwen/Qwen3.5-27B learning_rate=1e-5 max_tokens=512
```

Training metrics land in the recipe's `log_path` (under `/tmp/loops-cookbook/` by default) as `metrics.jsonl`; pass `wandb_project=...` to log to Weights & Biases.

## Async RL and off-policy bounds

In `rl/train_grpo_async.py`, `rl/train_math_async.py`, and `multiturn_rl/train_twenty_questions_async.py`, rollout workers generate trajectory groups continuously while the training loop consumes them — sampling never waits for the optimizer and vice versa. The cost is staleness: a rollout may have been sampled from a policy several optimizer steps old. Async mode pays off most when rollout wall-clock varies widely across a batch — multi-turn tasks, where each episode takes many sequential sampler calls, and long-form reasoning tasks like DeepMath, where solution length spans a wide range — since a synchronous loop would leave the trainer idle for the duration of the slowest rollout in every batch.

`max_steps_off_policy` bounds that staleness. Each trajectory group is tagged with the policy version it was sampled from; groups more than `max_steps_off_policy` steps behind the current trainer step are requeued instead of trained on. This rides on Loops' weight-versioning semantics:

- Every `optim_step` on the trainer bumps its **policy version**.
- After each step, the updated LoRA adapter is published to the paired sampler, which hot-swaps it in.
- Sampling requests carry an `X-Min-Policy-Version` floor — the sampler blocks until its live adapter reaches that version, so rollouts are never served by weights older than the version they were pinned to, and every sample result reports the policy version that produced it.

`max_steps_off_policy=2` is a reasonable default; raise it for more pipelining throughput, lower it to stay closer to on-policy.

`train_twenty_questions_async.py` and `train_math_async.py` also set `LOOPS_WARM_START_SAMPLER=true`, which makes the SDK provision the paired sampler alongside the trainer at run creation instead of at the first sampling request, so the two cold starts overlap. Leave it unset in train-only scripts (like the SFT recipe) so no sampler GPU is provisioned.

## Choosing a base model

Defaults here use Qwen3.5 models, which are supported by Loops trainers and samplers. Pick the smallest model that works for your task to keep iteration fast — `Qwen/Qwen3.5-4B` is a good starting point, and `Qwen/Qwen3-0.6B` trains in minutes if you just want to validate an environment before scaling up.

## Bringing a model tinker-cookbook doesn't know

Loops supports more base models than tinker-cookbook ships renderers for — `GET https://api.baseten.co/v1/loops/capabilities` (with your API key) lists what your workspace can train, including model families like GLM. To train one, define a `Renderer` that matches the model's chat template exactly, register it with `tinker_cookbook.renderers.register_renderer` (plus `tinker_cookbook.tokenizer_utils.register_tokenizer` when the weights repo ships no tokenizer files), and point the training config at your renderer name. [`sft/train_sft_glm.py`](sft/train_sft_glm.py) is the worked example — including a preflight that asserts the renderer reproduces `tokenizer.apply_chat_template` token-for-token, so template drift fails before a single GPU is provisioned. Larger MoE models like GLM-5.2 provision multi-node trainers: they take longer to become ready and bill more per hour, so deactivate the session as soon as you're done.

## Visualizing your rollouts

Use the `rollouts-dashboard` skill to visualize your rollouts as your training progresses. This skill helps your agents build helpful data visualizations for RL use cases.

## License and attribution

The recipes in this directory are adapted from [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), Copyright 2025 Thinking Machines Lab, licensed under the Apache License 2.0 (see [LICENSE](LICENSE) in this directory). Each adapted file carries an attribution header noting it was modified by Baseten. This Apache-2.0 licensing applies to the contents of `recipes/loops/`; the rest of this repository is MIT-licensed.
