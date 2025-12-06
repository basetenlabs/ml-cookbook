from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "axolotlai/axolotl:main-20251031-py3.11-cu128-2.8.0"

NUM_GPUS = 8
NUM_NODES = 1
GPU_TYPE = truss_config.Accelerator.H100

training_runtime = definitions.Runtime(
    start_commands=[
        f"torchrun --nproc-per-node={NUM_GPUS} train.py",
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(
            name="hf_access_token"
        ),  # The name of the HF Access Token secret in your B10 account
    },
    checkpointing_config=definitions.CheckpointingConfig(  # this defines BT_CHECKPOINT_DIR
        enabled=True,
    )
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=GPU_TYPE,
        count=NUM_GPUS,
    ),
    node_count=NUM_NODES,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name="Finetune Gemma 3 27B - ML Cookbook", job=training_job
)
