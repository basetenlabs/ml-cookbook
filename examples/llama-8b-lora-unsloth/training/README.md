# Llama 4 Maverick 17B LoRA Training with Unsloth

This example demonstrates how to fine-tune [Llama 4 Maverick 17B](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) using LoRA (Low-Rank Adaptation) with [Unsloth](https://github.com/unslothai/unsloth), which provides significant speedups for training.

**Resources:** 1 node, 1x H100 GPU

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   # pip
   pip install -U truss
   # or uv
   uv add truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples llama-8b-lora-unsloth
cd llama-8b-lora-unsloth
truss train push config.py
```
