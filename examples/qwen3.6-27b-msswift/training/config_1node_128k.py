from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.6-27B Long Context - ML Cookbook"
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

# Qwen3.6-27B is dense (no MoE), so EP doesn't help — TP is the natural
# 8-way split for the 27B weights across the node's 8 GPUs.
training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "EXP_TAG": "1n128k",
        "MAX_LENGTH": "131072",
        "TP": "8",
        "PP": "1",
        "EP": "1",
        "CP": "1",
        "RECOMPUTE_NUM_LAYERS": "4",
        "TRAIN_ITERS": "10",
    },
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
    cache_config=definitions.CacheConfig(enabled=True),
)

training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H200, count=8
    ),
)

my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

first_project_with_job = definitions.TrainingProject(
    name=project_name, job=my_training_job
)
