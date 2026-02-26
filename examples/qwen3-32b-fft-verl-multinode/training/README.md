# Qwen3 32B Full Fine-Tuning with VeRL (Multi-Node)

This example performs full fine-tuning of the Qwen3 32B model using the VeRL (Verifiable Reinforcement Learning) framework across multiple nodes on Baseten.

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
truss train init --examples qwen3-32b-fft-verl-multinode
cd qwen3-32b-fft-verl-multinode
truss train push config.py
```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
