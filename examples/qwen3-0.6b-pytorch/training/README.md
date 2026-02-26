# Qwen3 0.6B Fine-Tuning with PyTorch

This example fine-tunes the Qwen3 0.6B model using PyTorch on Baseten.

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
truss train init --examples qwen3-0.6b-pytorch
cd qwen3-0.6b-pytorch
truss train push config.py
```
