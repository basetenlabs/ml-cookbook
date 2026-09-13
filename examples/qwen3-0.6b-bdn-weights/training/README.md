# Qwen3 0.6B Fine-Tuning with BDN Weight Loading

This example fine-tunes the Qwen3 0.6B model using PyTorch on Baseten, with model weights loaded through [BDN](https://docs.baseten.co/training/concepts/storage#load-weights-and-data-with-bdn) instead of downloaded in the training script. BDN also supports loading training data from S3 (`s3://`), GCS (`gs://`), and R2 (`r2://`).

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
truss train init --examples qwen3-0.6b-bdn-weights
cd qwen3-0.6b-bdn-weights
truss train push config.py
```
