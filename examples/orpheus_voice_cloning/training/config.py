from truss_train import definitions
from truss.base import truss_config

GPU_COUNT = 1
# BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
BASE_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime"

runtime = definitions.Runtime(
    # enable the cache to skip model downloads on subsequent runs 
    # enable_cache=True,
    start_commands=[
        "apt-get update && apt-get install -y build-essential",
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
        "/bin/sh -c './run.sh'",
    ],
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
    },
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=GPU_COUNT,
    )
)

training_job = definitions.TrainingJob(
    compute=training_compute,
    runtime=runtime,
    image=definitions.Image(base_image=BASE_IMAGE),
)

training_project = definitions.TrainingProject(
    name="orpheus-tts training", 
    job=training_job
)