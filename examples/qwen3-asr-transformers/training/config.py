import os

from truss.base import truss_config
from truss_train import WeightsSource, definitions

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
GPU_COUNT = int(os.environ.get("GPU_COUNT", "1"))
if GPU_COUNT not in (1, 2, 4, 8):
    raise ValueError("GPU_COUNT must be one of: 1, 2, 4, 8")

DEFAULT_GRAD_ACC = 16 // GPU_COUNT
GRAD_ACC = int(os.environ.get("GRAD_ACC", str(DEFAULT_GRAD_ACC)))
if GRAD_ACC < 1:
    raise ValueError("GRAD_ACC must be a positive integer")

DEFAULT_PROJECT_NAME = "Qwen3-ASR-1.7B Finetuning (SFT)"
if GPU_COUNT > 1:
    DEFAULT_PROJECT_NAME += f" ({GPU_COUNT}x H100 DDP)"
PROJECT_NAME = os.environ.get("TRAINING_PROJECT_NAME", DEFAULT_PROJECT_NAME)

INIT_MODEL = "Qwen/Qwen3-ASR-1.7B"
INIT_MODEL_MOUNT = f"/app/models/{INIT_MODEL}"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        # Configure this secret in the Baseten workspace. It is used for the
        # dataset download and keeps this recipe compatible with gated data.
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        # The Qwen guide recommends limiting parallel FlashAttention build jobs
        # on machines with less than 96 GB RAM. Keep the same conservative cap
        # even though requirements.txt uses a prebuilt wheel.
        "MAX_JOBS": "4",
        # run.sh uses the pre-mounted model on Baseten and the Hub ID locally.
        "INIT_MODEL_PATH": INIT_MODEL_MOUNT,
        # Keep the global batch at 128 as the DDP process count changes:
        # 8 samples/device * accumulation * GPU_COUNT.
        "GRAD_ACC": str(GRAD_ACC),
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_compute = definitions.Compute(
    node_count=1,
    cpu_count=16,
    memory="96Gi",
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=GPU_COUNT,
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
    weights=[
        WeightsSource(
            source=f"hf://{INIT_MODEL}",
            mount_location=INIT_MODEL_MOUNT,
        ),
    ],
)

training_project = definitions.TrainingProject(
    name=PROJECT_NAME,
    job=training_job,
)
