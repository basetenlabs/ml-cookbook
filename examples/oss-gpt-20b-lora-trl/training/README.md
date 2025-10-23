# GPT-OSS 20B LoRA Training with TRL

This example demonstrates how to fine-tune [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) using LoRA (Low-Rank Adaptation) with [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft).

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
# for the oss-gpt lora example
truss train init --examples oss-gpt-20b-lora-trl && cd oss-gpt-20b-lora-trl
```

### Kick off the job

Make sure you've plugged in proper secrets (e.g. wandb api key, huggingface token) via Baseten Secrets and Environment Variables, and kick off your job

```bash
truss train push config.py
```

For more details, take a look at the [docs](https://docs.baseten.co/training/overview)
