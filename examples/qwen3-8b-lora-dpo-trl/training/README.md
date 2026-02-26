# Qwen3 8B LoRA DPO Fine-Tuning with TRL

This example fine-tunes the Qwen3 8B model using LoRA with Direct Preference Optimization (DPO) via the TRL framework on Baseten.

**Resources:** 1 node, 2x H100 GPUs

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples qwen3-8b-lora-dpo-trl
cd qwen3-8b-lora-dpo-trl
truss train push training/config.py
```
