# OSS GPT 20B Full Fine-Tuning with TRL

This example performs full fine-tuning of the OSS GPT 20B model using the TRL framework on Baseten.

**Resources:** 1 node, 8x H200 GPUs

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples oss-gpt-20b-fft-trl
cd oss-gpt-20b-fft-trl
truss train push config.py
```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
