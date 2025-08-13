from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = definitions.Runtime(
    start_commands = [
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"), # The name of the HF Access Token secret in your B10 account
        "HF_HOME": "/root/.cache/user_artifacts/hf_cache",
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"), # comment this out if you don't want to use wandb
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    )
)

training_compute = definitions.Compute(
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
    name="WhisperV3LargeTurbo common-accent",
    job=training_job
)
