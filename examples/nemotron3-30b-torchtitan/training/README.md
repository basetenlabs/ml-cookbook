# Nemotron-3 Nano 30B Training with TorchTitan

This example demonstrates how to fine-tune NVIDIA's Nemotron-3 Nano 30B model using TorchTitan on Baseten.

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
truss train init --examples nemotron3-30b-torchtitan
cd nemotron3-30b-torchtitan
truss train push config.py
```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.

## Serving the Model

To serve this model after training:

1. Generate the deployment config using the dry-run flag:
   ```bash
   truss train deploy_checkpoints --job-id=<your-job-id> --dry-run
   ```

2. Take the generated config and add the `--trust-remote-code` flag to the vLLM serve command.

3. Deploy using `truss push`:
   ```bash
   truss push
   ```

**Note:** Deploying from the UI does not currently work for this model. We are adding support for this soon.
