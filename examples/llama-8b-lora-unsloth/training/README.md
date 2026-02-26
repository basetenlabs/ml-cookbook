# Llama 4 Maverick 17B LoRA Training with Unsloth

This example demonstrates how to fine-tune [Llama 4 Maverick 17B](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) using LoRA (Low-Rank Adaptation) with [Unsloth](https://github.com/unslothai/unsloth), which provides significant speedups for training.

## Running the example

### Install `truss` 
Use the appropriate command for your package manager
```bash
# pip
pip install -U truss
# uv
uv add truss && uv sync --upgrade-package truss
```

### Create the workspace for your training project

```bash
# for the unsloth example
truss train init --examples llama-8b-lora-unsloth && cd llama-8b-lora-unsloth
```

### Kick off the job

Make sure you've plugged in proper secrets (e.g. wandb api key, huggingface token) via Baseten Secrets and Environment Variables, and kick off your job

```bash
truss train push config.py
```

For more details, take a look at the [docs](https://docs.baseten.co/training/overview)
