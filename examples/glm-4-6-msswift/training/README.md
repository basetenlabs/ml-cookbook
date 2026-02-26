# GLM-4-6 Fine-Tuning with MS-Swift

This example fine-tunes the GLM-4-6 model using the MS-Swift framework with MegatronLM on Baseten.

**Resources:** 2 nodes, 8x H200 GPUs each (16 GPUs total)

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
truss train init --examples glm-4-6-msswift
cd glm-4-6-msswift
truss train push config.py
```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
