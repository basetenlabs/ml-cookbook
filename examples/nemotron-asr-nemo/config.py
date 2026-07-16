from truss_train import definitions, WeightsSource
from truss.base import truss_config

# NeMo framework container ships NeMo + all ASR deps (kaldialign, sox, ffmpeg,
# libsndfile) and a matching PyTorch/CUDA stack. The Nemotron 3.5 ASR recipe
# requires NeMo 26.06+, so pin the 26.06 container (which also bundles the
# streaming-prompt fine-tune script + config).
BASE_IMAGE = "nvcr.io/nvidia/nemo:26.06"

# Base checkpoint. Delivered read-only via BDN before start commands run, so we
# never pay for the ~2.4GB download on billed GPU time (BDN mirrors + caches it
# across jobs). See https://docs.baseten.co/training/concepts/storage
INIT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
INIT_MODEL_MOUNT = f"/app/models/{INIT_MODEL}"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        # Gated on Hugging Face - used by BDN auth and any in-container downloads.
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        # run.sh reads this to locate the BDN-mounted checkpoint.
        "INIT_MODEL_MOUNT": INIT_MODEL_MOUNT,
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
    # Mount the base checkpoint via BDN instead of downloading it in run.sh.
    # `auth_secret_name` authenticates against the gated HF repo.
    weights=[
        WeightsSource(
            source=f"hf://{INIT_MODEL}",
            mount_location=INIT_MODEL_MOUNT,
            auth_secret_name="hf_access_token",
        ),
    ],
)

training_project = definitions.TrainingProject(
    name="Nemotron-3.5-ASR-Streaming Finetuned", job=training_job
)
