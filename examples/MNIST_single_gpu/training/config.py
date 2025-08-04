from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

training_runtime = definitions.Runtime(
    start_commands = [
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'",
    ],
    environment_variables={
    },
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
    cache_config=definitions.CacheConfig(
        enabled=True,
    )
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,  
        count=1,
    ),
    node_count=1
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)

training_project = definitions.TrainingProject(
    name="MNIST Training Job",
    job=training_job
) 