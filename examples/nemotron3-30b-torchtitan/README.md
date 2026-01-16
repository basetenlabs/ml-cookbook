# Nemotron-3 Nano 30B Training with TorchTitan

This example demonstrates how to fine-tune NVIDIA's Nemotron-3 Nano 30B model using TorchTitan.

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
