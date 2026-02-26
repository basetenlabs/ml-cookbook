from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

training_runtime = definitions.Runtime(
    start_commands=[
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'",
    ],
    checkpointing_config=definitions.CheckpointingConfig(  # this defines BT_CHECKPOINT_DIR
        enabled=True,
    ),
    cache_config=definitions.CacheConfig(  # this defines BT_RW_CACHE_DIR
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    cpu_count=4,
    memory="16Gi",
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name="MNIST Training Job", job=training_job
)
