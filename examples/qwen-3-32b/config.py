from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "axolotlai/axolotl:main-py3.11-cu126-2.7.1"

training_runtime = definitions.Runtime(
    start_commands = [
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'",
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"), # The name of the HF Access Token secret in your B10 account
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"), # The name of the WandB API Key secret in your B10 account
    },
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=False,
    ),
    cache_config=definitions.CacheConfig( # this defines BT_RW_CACHE_DIR
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,  
        count=8,
    ),
    node_count = 2
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)

training_project = definitions.TrainingProject(
    name="Qwen 3 32B Test",
    job=training_job
)
