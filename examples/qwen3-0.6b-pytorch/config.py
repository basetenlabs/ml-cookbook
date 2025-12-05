# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config

project_name = "demo/qwen3-0.6b"

# 1. Define a base image for your training job
BASE_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime"

# 2. Define the Runtime Environment for the Training Job
# This includes start commands and environment variables.a
# Secrets from the baseten workspace like API keys are referenced using
# `SecretReference`.

NUM_GPUS = 1

training_runtime = definitions.Runtime(
    start_commands=[
        "pip install transformers datasets accelerate bitsandbytes",
        f"torchrun --nproc-per-node={NUM_GPUS} train.py",
    ],
    environment_variables={
        # Secrets (ensure these are configured in your Baseten workspace)
        # Include other environment variables as needed
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
        enable_legacy_hf_mount=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
)

# 3. Define the Compute Resources for the Training Job
training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=1,
    ),
)

# 4. Define the Training Job
# This brings together the image, compute, and runtime configurations.
my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)


# This config will be pushed using the Truss CLI.
# The association of the job to the project happens at the time of push.
first_project_with_job = definitions.TrainingProject(
    name=project_name, job=my_training_job
)
