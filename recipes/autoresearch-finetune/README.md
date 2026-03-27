# Autoresearch Fine-Tuning on Baseten

Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) introduced a compelling division of labor: humans define what to optimize and what to hold constant, and an AI agent runs the experiments. This recipe applies that pattern to fine-tuning on Baseten.

With [Baseten Training](https://docs.baseten.co/training/overview), each experiment is a containerized job submitted via `truss train push`. The agent edits the training script, submits, monitors logs, reads the validation loss, and decides whether to keep or discard. No SSH, no environment management, no infra in the prompt.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/basetenlabs/ml-cookbook/main/images/autoresearch-darkmode.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/basetenlabs/ml-cookbook/main/images/autoresearch-lightmode.png">
  <img alt="Autoresearch fine-tuning loop diagram" src="https://raw.githubusercontent.com/basetenlabs/ml-cookbook/main/images/autoresearch-lightmode.png" width="600">
</picture>

The default configuration fine-tunes Qwen3-8B on [pirate-ultrachat-10k](https://huggingface.co/datasets/winglian/pirate-ultrachat-10k) using 2 H100 GPUs with [MS-Swift](https://github.com/modelscope/ms-swift)/Megatron, though Baseten Training is framework-agnostic. The same pattern works with Hugging Face Transformers, TRL, Axolotl, or plain PyTorch.

## Prerequisites

1. A [Baseten account](https://baseten.co/signup) with H100 GPU access.
2. The Truss CLI installed and configured (`pip install -U truss && truss login`).
3. A Hugging Face access token stored as a Baseten secret named `hf_access_token`. To create one, go to **Settings > Secrets** in the [Baseten dashboard](https://app.baseten.co) and add a secret with the name `hf_access_token` and your HF token as the value.
4. Git initialized in the working directory (the agent uses `git commit` and `git reset` to track experiments).

If you need H100 access or a higher GPU quota, [reach out to us](mailto:support@baseten.co).

## Getting started

Clone the repository and navigate to the recipe:

```bash
git clone https://github.com/basetenlabs/ml-cookbook.git
cd ml-cookbook/recipes/autoresearch-finetune
```

Review `training/config.py` and edit the constants at the top to match your model, dataset, and GPU allocation:

```python
MODEL = "Qwen/Qwen3-8B"                          # Any HuggingFace model path
DATASET = "winglian/pirate-ultrachat-10k"         # Any HuggingFace dataset
EVAL_SPLIT_RATIO = "0.01"                         # Fraction held out for validation
ACCELERATOR = "H100"                              # GPU type: H100, H200, A100
GPU_COUNT = 2                                     # GPUs per training job
```

If you're using [Claude Code](https://docs.anthropic.com/en/docs/claude-code), you can add the Baseten docs MCP server so the agent can look up Truss CLI and Training API docs on the fly:

```bash
claude mcp add --transport http baseten-docs https://docs.baseten.co/mcp
```

Then point your agent at this directory and tell it to read `prompt.md`:

```
Read prompt.md and optimize val_loss.
```

The agent picks up the instructions from `prompt.md`, which tells it how to submit and monitor experiments via the Truss CLI. It edits `training/run.sh` to change hyperparameters (LoRA rank, learning rate, batch size, etc.), submits each experiment to Baseten, and hill-climbs on validation loss. Improvements are kept as git commits; regressions are discarded via `git reset`.

A successful experiment produces output like this:

```
---
val_loss:         1.389524
total_seconds:    382
peak_vram_mb:     23715
```

## Changing the model or dataset

Edit the constants at the top of `training/config.py`. Set `MODEL` to any Hugging Face model path and `DATASET` to any HuggingFace dataset that MS-Swift can consume. Adjust `EVAL_SPLIT_RATIO` if you want a larger or smaller validation set (the default 0.01 holds out 1% of the data).

If MS-Swift cannot auto-detect the model architecture (which can happen with very new or custom models), add a `--model_type` flag to the megatron command in `training/run.sh`. You can find valid model types in the [MS-Swift documentation](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md).

## Serving your best checkpoint

Every training job saves its checkpoint to `$BT_CHECKPOINT_DIR/lora-output`. Once you're happy with the results, deploy it as an inference endpoint:

```bash
truss train deploy_checkpoints --job-id <job_id>
```

Use `--dry-run` to inspect the deployment config before pushing. For more advanced workflows, see [serving trained models](https://docs.baseten.co/training/deployment).

## File reference

- **`prompt.md`** — the agent's instruction set. Tells it what to optimize, how to submit and monitor experiments, and how to track results with git. This is the source of truth for agent behavior.
- **`training/run.sh`** — the training script the agent edits. Contains the `megatron sft` command with all hyperparameters inline. The results parsing block at the end (below the "DO NOT EDIT" line) extracts `val_loss`, `total_seconds`, and `peak_vram_mb` from the logs.
- **`training/config.py`** — defines the Baseten training job: base image, model weights, GPU allocation, and environment variables. Edit the constants at the top for your setup; the rest generally doesn't need to change.

## Private HF Model

If you'd like to use a private huggingface model, make sure to include add a huggingface token to your Baseten secrets as `hf_access_token` and include `"HF_TOKEN": definitions.SecretReference("hf_access_token")` in your `environment_variables` within your `config.py`
