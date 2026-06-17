from truss_train import definitions
from truss.base import truss_config

# Fine-tune NVIDIA Isaac-GR00T (a ~3B Vision-Language-Action model for robots)
# on a LeRobot-format dataset. Training is pure PyTorch on pre-recorded
# demonstrations, so no simulator or rendering stack is involved.
PROJECT_NAME = "isaac-groot-n1-finetune"

# Isaac-GR00T requires Python 3.10. This public base image ships 3.11, so run.sh
# uses `uv` (GR00T's documented env manager) to build a 3.10 environment.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"

training_runtime = definitions.Runtime(
    start_commands=["/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"],
    environment_variables={
        # Required only for gated HF datasets/models. Create a workspace secret
        # named `hf_access_token`. The public GR00T base model needs no token.
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
    },
    cache_config=definitions.CacheConfig(enabled=True, enable_legacy_hf_mount=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        # Full fine-tune needs a 40GB+ GPU; L40S (48GB) fits, H100 is faster. On a
        # 24GB GPU (A10G/L4), switch run.sh to LoRA or --no-tune_diffusion_model.
        accelerator=truss_config.Accelerator.L40S,
        count=1,
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
