# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config

# 1. Define a base image for your training job. You can also use
# private images via AWS IAM or GCP Service Account authentication.
BASE_IMAGE = "axolotlai/axolotl:main-py3.11-cu128-2.7.1"

# 2. Define the Runtime Environment for the Training Job
# This includes start commands and environment variables.
# Secrets from the baseten workspace like API keys are referenced using 
# `SecretReference`.
training_runtime = definitions.Runtime(
    start_commands=[ # Example: list of commands to run your training script
        # "pip install -r requirements.txt", # pip install requirements on top of base image
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'",  
    ],
    environment_variables={
        # Secrets (ensure these are configured in your Baseten workspace)
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "WANDB_API_KEY" : definitions.SecretReference(name="wandb_api_key"),
        "HELLO": "WORLD"
    },
    checkpointing_config=definitions.CheckpointingConfig( # this defines BT_CHECKPOINT_DIR
        enabled=True,
    ),
)

# 3. Define the Compute Resources for the Training Job
training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,  
        count=4,  
    ),
)

# 4. Define the Training Job
# This brings together the image, compute, and runtime configurations.
training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)


# This config will be pushed using the Truss CLI.
# The association of the job to the project happens at the time of push.
training_project = definitions.TrainingProject(
    name="LoRA Training Job - llama-8b",
    job=training_job
)