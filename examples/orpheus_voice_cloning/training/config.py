from truss_train import definitions
from truss.base import truss_config

GPU_COUNT = 1

runtime = definitions.Runtime(
    # enable the cache to skip model downloads on subsequent runs 
    # enable_cache=True,
    start_commands=[
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
        "/bin/sh -c './run.sh'",
    ],
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_access_token"),
        # Shared configuration used across multiple processes
        # "BASE_MODEL_ID": "canopylabs/orpheus-3b-0.1-pretrained", # this is a huggingface model id
    },
)

training_job = definitions.TrainingJob(
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H100,
            count=GPU_COUNT,
        )
    ),
    runtime=runtime,
    # image=definitions.Image(base_image="nvcr.io/nvidia/pytorch:25.05-py3"),
    image=definitions.Image(base_image="python:3.11.13"),
)

first_project = definitions.TrainingProject(name="orpheus-voice_cloning", job=training_job)
