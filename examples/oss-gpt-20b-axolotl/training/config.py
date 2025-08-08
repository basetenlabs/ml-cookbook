# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config
from truss_train.definitions import CheckpointingConfig

project_name = "Axolotl Training Job - gpt-oss-20b"

# 1. Define a base image for your training job
# must use torch 2.7.0 for vllm
BASE_IMAGE = "axolotlai/axolotl:main-py3.11-cu126-2.7.1"

# 2. Define the Runtime Environment for the Training Job
# This includes start commands and environment variables.a
# Secrets from the baseten workspace like API keys are referenced using
# `SecretReference`.

# checkpoint_mount_dir = "/workspace/checkpoints"
# checkpointing = CheckpointingConfig(enabled=True, checkpoint_path=checkpoint_mount_dir)

training_runtime = definitions.Runtime(
    start_commands=[ # Example: list of commands to run your training script
        "pip install -U accelerate",
        # "git fetch && git checkout accelerate-cp",
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"
    ],
    environment_variables={
        # Secrets (ensure these are configured in your Baseten workspace)
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "WANDB_API_KEY" : definitions.SecretReference(name="wandb_api_key"),  # Optional: comment this out if you don't want to use wandb
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    )
)

# 3. Define the Compute Resources for the Training Job
training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=8,
    ),
)

# 4. Define the Training Job
# This brings together the image, compute, and runtime configurations.
my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)


# This config will be pushed using the Truss CLI.
# The association of the job to the project happens at the time of push.
first_project_with_job = definitions.TrainingProject(
    name=project_name,
    job=my_training_job
)