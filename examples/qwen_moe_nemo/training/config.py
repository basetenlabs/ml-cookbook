from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "nvcr.io/nvidia/nemo:25.07"

training_runtime = definitions.Runtime(
    start_commands=[
        "./run.sh"
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # "HF_HOME": "/root/.cache/user_artifacts/hf_cache",
        # "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
    },
    cache_config=definitions.CacheConfig(
        enabled=False,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
        checkpoint_path="/tmp/training_checkpoints",
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
    runtime=training_runtime
)

training_project = definitions.TrainingProject(
    name="Nemo template",
    job=training_job
)