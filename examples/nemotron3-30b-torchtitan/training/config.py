from truss.base import truss_config
from truss_train import definitions

BASE_IMAGE = "ghcr.io/pytorch/pytorch-nightly:71739c4-cu12.9.1"
PROJECT_NAME = "Nemotron3-Nano"


ACCELERATOR = truss_config.Accelerator.H200
NGPU = 8
NODE_COUNT = 1

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh && ./run.sh",
    ],
    environment_variables={
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config = definitions.CheckpointingConfig(
        enabled=True,
    )
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=ACCELERATOR,
        count=NGPU,
    ),
    node_count=NODE_COUNT,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
