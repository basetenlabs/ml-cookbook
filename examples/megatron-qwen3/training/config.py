from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "baseten/megatron:0.0.1"
PROJECT_NAME = "Megatron Qwen3 H200"
NODE_COUNT = 2
MICRO_BATCH_SIZE = 1
NUM_GPUS = 8

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(
            name="bt_hf_access_token"
        ),  # The name of the HF Access Token secret in your B10 account
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        "WANDB_API_KEY": definitions.SecretReference(
            name="wandb_api_key"
        ),  # comment this out if you don't want to use wandb
        "GLOBAL_BATCH_SIZE": str(MICRO_BATCH_SIZE * NUM_GPUS * NODE_COUNT),
        "MICRO_BATCH_SIZE": str(MICRO_BATCH_SIZE),
        "SHOULD_CLEAR_CACHE": "false",
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
        accelerator=truss_config.Accelerator.H200,
        count=8,
    ),
    node_count=NODE_COUNT,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
