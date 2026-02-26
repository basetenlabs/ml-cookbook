# Gemma 27B Fine-Tuning with Axolotl

This example fine-tunes Google's Gemma 27B model using the Axolotl framework on Baseten.

**Resources:** 2 nodes, 8x H100 GPUs each (16 GPUs total)

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
truss train init --examples gemma-27b-axolotl
cd gemma-27b-axolotl
truss train push config.py
```

> **Note:** This example requires more than 4x H100 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
