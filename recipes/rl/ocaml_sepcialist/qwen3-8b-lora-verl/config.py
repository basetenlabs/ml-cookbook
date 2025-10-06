# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config

# Project name
project_name = "OCaml Specialist - veRL GRPO"

# Base image with VERL dependencies
# You may need to build a custom image with VERL installed
BASE_IMAGE = "verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0-te2.2"

# Define the Runtime Environment for the Training Job
training_runtime = definitions.Runtime(
    start_commands=[
        "git clone https://github.com/volcengine/verl && cd verl",
        "pip3 install --no-deps -e .",
        "cd ..",
        "apt-get install ocaml -y",
        # Make run script executable
        "chmod +x ./run.sh",
        # Run the training script
        "./run.sh",
    ],
    environment_variables={
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
    },
    # Enable training cache for faster iteration
    enable_cache=False,
    checkpointing_config = definitions.CheckpointingConfig(
        enabled=True,
        checkpoint_path="/tmp/checkpoints"
    )
)

# Define the Compute Resources for the Training Job
training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=4,  
    ),
    node_count=1,  
)

# Define the Training Job
my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

# Create the training project
verl_training_project = definitions.TrainingProject(
    name=project_name,
    job=my_training_job
)