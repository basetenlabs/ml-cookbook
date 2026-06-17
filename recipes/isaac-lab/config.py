from truss_train import definitions
from truss.base import truss_config

project_name = "isaac-lab-cartpole"

# Ubuntu 22.04 / GLIBC 2.35 / Python 3.11 / torch 2.7.0+cu128 — an exact match for
# Isaac Sim 5.1's requirements, so no Python or torch reinstall is needed.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"

# Isaac Sim's RTX renderer initializes Vulkan even in --headless mode and requires
# hardware ray tracing. This only works on RT-core GPUs (T4, A10G, L4). Compute SKUs
# (A100, H100, B200) have no RT cores and cannot create a Vulkan device — Isaac Sim
# fails at GPU-foundation init on those. A10G is a good default for Isaac Lab RL.
ACCELERATOR = truss_config.Accelerator.A10G
GPU_COUNT = 1

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh",
        "./run.sh",
    ],
    environment_variables={
        # Default is "compute,utility", which injects only CUDA libs. "all" makes the
        # NVIDIA container runtime also inject the graphics/Vulkan stack (libGLX_nvidia,
        # nvidia_icd.json, libnvidia-rtcore) that the RTX renderer needs.
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        # Accept the Omniverse EULA non-interactively; otherwise `import isaacsim` blocks.
        "OMNI_KIT_ACCEPT_EULA": "YES",
        # Route pip + Kit shader caches onto the persistent cache mount for fast reruns.
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
