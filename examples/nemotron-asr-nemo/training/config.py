from truss_train import definitions
from truss.base import truss_config

# NeMo framework container ships with all ASR system dependencies
# (sox, ffmpeg, libsndfile) and a compatible PyTorch/CUDA stack.
BASE_IMAGE = "nvcr.io/nvidia/nemo:25.09"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        # Gated on Hugging Face - required to download the base checkpoint.
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
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
)

training_project = definitions.TrainingProject(
    name="Nemotron-3.5-ASR-Streaming Finetuned", job=training_job
)
