from truss_train import definitions
from truss.base import truss_config

project_name = "Isaac Lab Cartpole RL - ML Cookbook"

# Matches Isaac Sim 5.1 exactly (Ubuntu 22.04, Python 3.11, torch 2.7.0+cu128).
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"

# Isaac Sim's RTX renderer needs hardware ray tracing, so it only runs on RT-core GPUs
# (T4, A10G, L4) — not on compute SKUs (A100, H100, B200), which can't create a Vulkan device.
ACCELERATOR = truss_config.Accelerator.A10G
GPU_COUNT = 1

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh",
        "./run.sh",
    ],
    environment_variables={
        # "all" makes the container runtime inject the graphics/Vulkan stack, not just CUDA.
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        # Accept the Omniverse EULA non-interactively; otherwise `import isaacsim` blocks.
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PIP_CACHE_DIR": "/root/.cache/pip",
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
        checkpoint_path="/tmp/checkpoints",
    ),
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=ACCELERATOR,
        count=GPU_COUNT,
    ),
    node_count=1,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

_ = definitions.TrainingProject(
    name=project_name,
    job=training_job,
)
