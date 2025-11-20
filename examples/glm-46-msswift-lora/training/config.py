from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "baseten/megatron:0.0.7"
PROJECT_NAME = "Megatron-GLM-4.6-MSSwift-LORA multinode"

NUM_NODES = 2
ITERATIVE = True
file = "william_run.sh" if not ITERATIVE else "rssh.sh"

training_runtime = definitions.Runtime(
    start_commands=[f"/bin/sh -c 'chmod +x ./{file} && ./{file}'"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(
            name="hf_access_token"
        ),  # The name of the HF Access Token secret in your B10 account
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        "WANDB_API_KEY": definitions.SecretReference(
            name="wandb_api_key"
        ),  # comment this out if you don't want to use wandb
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
    node_count=NUM_NODES,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
