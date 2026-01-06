# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config

project_name = "baseten-admin/eagle-3-head"

# 1. Define a base image for your training job
BASE_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime"

# 2. Define the Runtime Environment for the Training Job
# This includes start commands and environment variables.a
# Secrets from the baseten workspace like API keys are referenced using
# `SecretReference`.

training_runtime = definitions.Runtime(
    start_commands=[  # Example: list of commands to run your training script
        "apt-get update",
        "apt-get install -y --no-install-recommends build-essential",
        "conda install -y -c nvidia cuda-toolkit",
        "apt install -y numactl",
        "pip install transformers datasets accelerate bitsandbytes",
        "/bin/sh -c 'chmod +x ./rssh.sh && ./rssh.sh'"
    ],
    environment_variables={
        # Secrets (ensure these are configured in your Baseten workspace)
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
        # Include other environment variables as needed
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
        enable_legacy_hf_mount=True,
        require_cache_affinity=False,
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
        count=4,
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
