# Llama 3.1 8B LoRA Training with Unsloth

This example demonstrates how to fine-tune [Llama 3.1 8B](https://huggingface.co/Meta-Llama-3.1-8B-Instruct) using LoRA (Low-Rank Adaptation) with [Unsloth](https://github.com/unslothai/unsloth), which provides significant speedups for training.

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

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)
2. Accept the terms and request access to the model (if it's gated)

### Kick off the job

Make sure you've plugged in proper secrets (e.g. wandb api key, huggingface token) via Baseten Secrets and Environment Variables, and kick off your job

```bash
truss train push config.py
```

For more details, take a look at the [docs](https://docs.baseten.co/training/overview)
