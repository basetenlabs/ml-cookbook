from truss_train import definitions, WeightsSource
from truss.base.truss_config import AcceleratorSpec

# ---------------------------------------------------------------------------
# Edit these before running
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen3-8B"                          # Any HuggingFace model path
DATASET = "winglian/pirate-ultrachat-10k"         # Any HuggingFace dataset
EVAL_SPLIT_RATIO = "0.01"                         # Fraction held out for validation
ACCELERATOR = "H100"                              # GPU type: H100, H200, A100
GPU_COUNT = 2                                     # GPUs per training job

# ---------------------------------------------------------------------------
# Training job definition (generally don't need to edit below)
# ---------------------------------------------------------------------------

BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"
MODEL_MOUNT = f"/app/models/{MODEL}"

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "MODEL": MODEL_MOUNT,
        "DATASET": DATASET,
        "EVAL_SPLIT_RATIO": EVAL_SPLIT_RATIO,
    },
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_compute = definitions.Compute(
    accelerator=AcceleratorSpec(accelerator=ACCELERATOR, count=GPU_COUNT),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
    weights=[
        WeightsSource(
            source=f"hf://{MODEL}",
            mount_location=MODEL_MOUNT,
        ),
    ],
)

training_project = definitions.TrainingProject(
    name="autoresearch-finetune",
    job=training_job,
)
