from truss_train import definitions, WeightsSource
from truss.base import truss_config

# Use the `-devel` image (not `-runtime`): the default flex_attention backend
# compiles its kernels at runtime via torch.compile/Inductor/Triton, which
# needs a host C compiler (gcc) + CUDA dev headers. The slim runtime image
# ships neither and fails with "Failed to find C compiler". If you'd rather
# avoid runtime compilation entirely, switch run.sh to the SDPA training
# config and the runtime image is sufficient.
BASE_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
PROJECT_NAME = "OmniVoice Finetuning (TTS)"

# Pretrained OmniVoice checkpoint to fine-tune from. We mount it so the job
# doesn't re-download it from the Hub on every run; run.sh consumes
# INIT_FROM_CHECKPOINT to point the training config at this directory.
INIT_MODEL = "k2-fsa/OmniVoice"
INIT_MODEL_MOUNT = f"/app/models/{INIT_MODEL}"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        # Specify the HF token secret in your baseten workspace for accessing
        # your dataset (and the pretrained model if it is gated).
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        # run.sh rewrites the training config's `init_from_checkpoint` to this
        # mounted path; locally it is unset and falls back to the HF repo id.
        "INIT_FROM_CHECKPOINT": INIT_MODEL_MOUNT,
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
