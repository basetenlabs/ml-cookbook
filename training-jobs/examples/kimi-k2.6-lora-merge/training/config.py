from truss_train import (
    CheckpointingConfig,
    Compute,
    Image,
    LoadCheckpointConfig,
    LoopsCheckpoint,
    Runtime,
    TrainingJob,
    TrainingProject,
    WeightsSource,
)
from truss.base.truss_config import AcceleratorSpec

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

# The Loops run + checkpoint to merge. target="sampler" selects the exported
# inference adapter (the deployable LoRA), not the full trainer state. Fill these
# in with your own run — find them via `truss train checkpoints list`.
LOOPS_RUN_ID = "<your-loops-run-id>"
LOOPS_CHECKPOINT_NAME = "<your-sampler-checkpoint-name>"

training_runtime = Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    # The merged checkpoint is written to $BT_CHECKPOINT_DIR; the Kimi-K2.6
    # base is ~555 GB, so size the volume accordingly.
    checkpointing_config=CheckpointingConfig(enabled=True, volume_size_gib=700),
    # Pull the Loops sampler adapter via the user-facing checkpoint loader
    # (resolves to bt://loops:... server-side — no raw S3 URL needed). It lands
    # under /tmp/loaded_checkpoints/.
    load_checkpoint_config=LoadCheckpointConfig(
        enabled=True,
        checkpoints=[
            LoopsCheckpoint.from_checkpoint(
                run_id=LOOPS_RUN_ID,
                checkpoint_name=LOOPS_CHECKPOINT_NAME,
                target="sampler",
            ),
        ],
    ),
)

# H100 is used for its large node-local disk (the 555 GB base mirror), not the
# GPU — the merge runs on CPU. The merge peaks at ~50 GB RAM / a few cores.
training_compute = Compute(
    accelerator=AcceleratorSpec(accelerator="H100", count=1),
    cpu_count=16,
    memory="96Gi",
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
    # The quantized Kimi-K2.6 base, mirrored to /app/base.
    weights=[
        WeightsSource(
            source="hf://moonshotai/Kimi-K2.6",
            mount_location="/app/base",
            auth_secret_name="hf_access_token",
        ),
    ],
)

training_project = TrainingProject(
    name="kimi-k2.6-lora-merge",
    job=training_job,
)
