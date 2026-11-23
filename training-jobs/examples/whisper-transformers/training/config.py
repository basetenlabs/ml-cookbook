import os

from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

environment_variables: dict[str, str | definitions.SecretReference] = {}
if hub_model_id := os.getenv("HF_HUB_MODEL_ID"):
    environment_variables["HF_HUB_MODEL_ID"] = hub_model_id
if secret_name := os.getenv("HF_TOKEN_SECRET_NAME"):
    environment_variables["HF_TOKEN"] = definitions.SecretReference(name=secret_name)
elif hub_model_id:
    raise ValueError("Set HF_TOKEN_SECRET_NAME before exporting to Hugging Face.")

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables=environment_variables,
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
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name="WhisperV3LargeTurbo common-accent", job=training_job
)
