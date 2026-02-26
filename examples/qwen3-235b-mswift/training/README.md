# Qwen3 235B LoRA Fine-Tuning with MS-Swift

This example fine-tunes the Qwen3 235B model using LoRA with the MS-Swift framework and MegatronLM on Baseten.

**Resources:** 2 nodes, 8x H200 GPUs each (16 GPUs total)

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples qwen3-235b-mswift
cd qwen3-235b-mswift
truss train push training/config.py
```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
