from truss_train import definitions
from truss.base import truss_config

project_name = "minimax-2-5-ori-new2"

# Prebuilt image with Megatron + ms-swift.
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

training_runtime = definitions.Runtime(
    start_commands=[
        "/bin/bash -c 'chmod +x ./rssh.sh && ./rssh.sh'",
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
        # Override in Baseten UI for your run.
        "MODEL_ID": "MiniMaxAI/MiniMax-M2.5",
        "DATASET_ID": "winglian/pirate-ultrachat-10k",
        "CHECKPOINT_NAME": "minimax-2-5-lora-8-32",
        "SAVE_FULL_MODEL": "false",
        "TRAIN_TYPE": "lora",
        "LR_WARMUP_FRACTION": "0.05",
        "MIN_LR": "1e-5",
        "EVAL_INTERVAL": "5",
        "MAX_LENGTH": "1024",
        "NUM_WORKERS": "8",
        "DATASET_NUM_PROC": "8",
        "MASTER_PORT": "29500",
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
        enable_legacy_hf_mount=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

# Multi-node distributed job for large MiniMax 2.5 tuning.
training_compute = definitions.Compute(
    node_count=2,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H200,
        count=8,
    ),
)

my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

first_project_with_job = definitions.TrainingProject(name=project_name, job=my_training_job)
