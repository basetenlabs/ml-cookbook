# Qwen3 30B Fine-Tuning with MS-Swift (Multi-Node)

This example fine-tunes the Qwen3 30B model using the MS-Swift framework with MegatronLM across multiple nodes on Baseten.

**Resources:** 2 nodes, 8x H100 GPUs each (16 GPUs total)

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples qwen3-30b-mswift-multinode
cd qwen3-30b-mswift-multinode
truss train push training/config.py
```

> **Note:** This example requires more than 4x H100 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
