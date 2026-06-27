# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config

project_name = "LoRA GLM-5.1 - ML Cookbook"

# 1. Define a base image for your training job
# GLM-5.1 (glm_moe_dsa) requires megatron-core >= 0.17.1: earlier builds lack the
# DSA indexer kernel and reject the model's interleaved-RoPE + multi-latent-attention combo.
BASE_IMAGE = "baseten/megatron:py3.12.13-cuda12.8.1-torch2.9.1-fa2.8.3-megatron0.17.1-msswift4.3.1-peftstamp"

# 2. Define the Runtime Environment for the Training Job
training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh && ./run.sh"
    ],
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
)

# 3. Define the Compute Resources for the Training Job
# 256 routed experts shard cleanly only across power-of-2 GPU counts; 4x8 gives EP=32.
training_compute = definitions.Compute(
    node_count=4,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H200,
        count=8,
    ),
)

# 4. Define the Training Job
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
