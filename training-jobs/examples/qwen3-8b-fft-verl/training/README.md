# Qwen3 8B Full Fine-Tuning with VeRL

This example performs full fine-tuning of the Qwen3 8B model using the VeRL (Verifiable Reinforcement Learning) framework on Baseten.

**Resources:** 1 node, 8x H100 GPUs

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
truss train init --examples qwen3-8b-fft-verl
cd qwen3-8b-fft-verl
truss train push config.py
```

> **Note:** This example requires more than 4x H100 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
