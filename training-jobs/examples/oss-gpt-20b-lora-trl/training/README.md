# OSS GPT 20B LoRA Fine-Tuning with TRL

This example fine-tunes the OSS GPT 20B model using LoRA with the TRL framework on Baseten.

**Resources:** 1 node, 2x H100 GPUs

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
truss train init --examples oss-gpt-20b-lora-trl
cd oss-gpt-20b-lora-trl
truss train push config.py
```
