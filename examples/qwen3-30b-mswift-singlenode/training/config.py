from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"
PROJECT_NAME = "Qwen3-30B Megatron Single Node - ML Cookbook"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(
            name="hf_access_token"
        ),  # The name of the HF Access Token secret in your B10 account
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=8,
    ),
    node_count=1,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
