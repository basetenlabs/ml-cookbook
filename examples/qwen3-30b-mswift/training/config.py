from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "baseten/megatron:0.0.3"
PROJECT_NAME = "MSwift Megatron - Single Node"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run_1node.sh && ./run_1node.sh'"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(
            name="hf_access_token"
        ),  # The name of the HF Access Token secret in your B10 account
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
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
