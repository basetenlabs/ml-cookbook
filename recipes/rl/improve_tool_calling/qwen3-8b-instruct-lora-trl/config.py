# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config

# Project name
project_name = "Tool Call qwen3 8b lora trl grpo"

# Base image with VERL dependencies
# You may need to build a custom image with VERL installed
BASE_IMAGE = "verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0-te2.2"

# Define the Runtime Environment for the Training Job
training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh",
        "./run.sh",
    ],
    environment_variables={
        "HF_ACCESS_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
        "WANDB_PROJECT": "Qwen3-8B-Tool-Calling-GRPO",
    },
    # Enable training cache for faster iteration
    cache_config = definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config = definitions.CheckpointingConfig(
        enabled=True,
        checkpoint_path="/tmp/checkpoints"
    )
)

# Define the Compute Resources for the Training Job
training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=8,
    ),
    node_count=1,  
)

# Define the Training Job
training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

# Create the training project
_ = definitions.TrainingProject(
    name=project_name,
    job=training_job
)
