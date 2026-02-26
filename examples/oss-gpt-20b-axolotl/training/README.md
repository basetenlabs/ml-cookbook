# OSS GPT 20B Fine-Tuning with Axolotl

This example fine-tunes the OSS GPT 20B model using the Axolotl framework on Baseten.

**Resources:** 1 node, 4x H100 GPUs

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples oss-gpt-20b-axolotl
cd oss-gpt-20b-axolotl
truss train push config.py
```
