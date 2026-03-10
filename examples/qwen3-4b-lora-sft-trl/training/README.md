# Qwen3-4B LoRA SFT with TRL

Fine-tunes Qwen3-4B with LoRA using TRL's SFTTrainer on Baseten. Uses the pirate-ultrachat-10k dataset, which teaches the model to respond in pirate dialect.

**Resources:** 1x H100 GPU

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   # pip
   pip install -U truss
   # or uv
   uv add truss
   ```

## Getting started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples qwen3-4b-lora-sft-trl
cd qwen3-4b-lora-sft-trl
truss train push config.py
```

## Deploy

After training completes, deploy a checkpoint:

```bash
truss train deploy_checkpoints
```

Follow the interactive prompts to select a checkpoint, name your model, and choose a GPU. This step requires `hf_access_token` in [Baseten Secrets](https://app.baseten.co/settings/secrets).

## Details

- **Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- **Dataset:** [winglian/pirate-ultrachat-10k](https://huggingface.co/datasets/winglian/pirate-ultrachat-10k)
- **Method:** LoRA (rank 8) with SFTTrainer
- **Training time:** ~2 minutes on 1x H100
- **Steps:** 50 (checkpoints at 25 and 50)

For a full walkthrough, see the [Getting started tutorial](https://docs.baseten.co/training/getting-started).
