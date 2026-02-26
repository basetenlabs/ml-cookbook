# GLM-4.7 Flash Fine-Tuning with MS-Swift

This example fine-tunes the GLM-4.7 Flash model using the MS-Swift framework with MegatronLM on Baseten.

**Resources:** 1 node, 8x H100 GPUs

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples glm-4.7-flash-msswift
cd glm-4.7-flash-msswift
truss train push training/config.py
```

> **Note:** This example requires more than 4x H100 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
