# Llama 3.1 8B LoRA Fine-Tuning with Unsloth

This example fine-tunes Meta's Llama 3.1 8B Instruct model using LoRA with the Unsloth framework on Baseten.

**Resources:** 1 node, 1x H100 GPU

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples llama-8b-lora-unsloth
cd llama-8b-lora-unsloth
truss train push config.py
```
