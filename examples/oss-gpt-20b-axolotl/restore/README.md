## Run instructions

This example demonstrates how to restore a checkpoint from a prior training job and resume training from that checkpoint.

### Update config.py
- **Load Checkpoint Configuration**: Update the `load_checkpoint_config` section to specify which checkpoint to restore:
  - `job_id`: The job ID of the previous training job (found in the UI or from `truss train view --job-id=<previous-job-id>`)
  - `checkpoint_name`: The name of the checkpoint to load (e.g., `"checkpoint-14"`)
  - Example:
    ```python
    load_checkpoint_config=definitions.LoadCheckpointConfig(
        enabled=True,
        checkpoints=[definitions.BasetenCheckpoint.from_named_checkpoint(job_id="4q9g403", checkpoint_name="checkpoint-14")],
        download_folder="/tmp/loaded_checkpoints",
    ),
    ```

### Update config.yaml
- **Resume from Checkpoint**: Set the `resume_from_checkpoint` field to the checkpoint path. You can find the correct checkpoint path in one of two ways:
  1. From the UI: Navigate to your previous training job and locate the checkpoint path
  2. From CLI: Run `truss train view --job-id=<previous-job-id>` to see checkpoint details
  - The path format should be: `/tmp/loaded_checkpoints/<old-job-id>/rank-0/<checkpoint-name>/`
  - Set: `resume_from_checkpoint: /tmp/loaded_checkpoints/4q9g403/rank-0/checkpoint-14/` inside your Axolotl `config.yaml`

### Launch run 

```
truss train push config.py --job-name="RestartFromCheckpoints"
```