# Orpheus Fine-Tuning with Transformers

This example fine-tunes the Orpheus audio model using the HuggingFace Transformers framework on Baseten.

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
truss train init --examples orpheus-transformers
cd orpheus-transformers
truss train push config.py
```
