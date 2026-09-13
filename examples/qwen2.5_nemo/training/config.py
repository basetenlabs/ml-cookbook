from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "nvcr.io/nvidia/nemo:25.07"

training_runtime = definitions.Runtime(
    start_commands=[
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
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
    runtime=training_runtime
)

training_project = definitions.TrainingProject(
    name="Nemo-qwen2.5-nemo 1node",
    job=training_job
)