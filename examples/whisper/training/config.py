from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = definitions.Runtime(
    start_commands = [
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"), # The name of the HF Access Token secret in your B10 account
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"), # comment this out if you don't want to use wandb
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=8,
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)

training_project = definitions.TrainingProject(
    name="WhisperV3LargeTurbo common-accent testrun",
    job=training_job
)
