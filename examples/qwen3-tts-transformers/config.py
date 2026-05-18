from truss_train import definitions, WeightsSource
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
PROJECT_NAME = "Qwen3-TTS-12Hz Finetuning (SFT)"

INIT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
INIT_MODEL_MOUNT = f"/app/models/{INIT_MODEL}"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        # Specify the HF token secret in your baseten workspace for accessing your dataset.
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        # run.sh consumes this; defaults to the HF repo id for local dev.
        "INIT_MODEL_PATH": INIT_MODEL_MOUNT,
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=1,
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
    name=PROJECT_NAME, job=training_job
)
