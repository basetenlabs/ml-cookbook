from truss_train import definitions
from truss.base import truss_config

project_name = "isaac-groot-n1-finetune"

# NVIDIA Isaac-GR00T is a ~3B Vision-Language-Action model for humanoid robots.
# Fine-tuning runs on pre-recorded LeRobot datasets with plain PyTorch — no
# simulator, Omniverse, or rendering stack — so it works on any GPU (no RT-core
# requirement, unlike the Isaac Lab recipe).
#
# GR00T pins Python 3.10; this public image ships 3.11, so run.sh uses `uv` to
# build a 3.10 environment.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"

# A10G (24GB): run.sh passes --no-tune_diffusion_model so the fine-tune fits in
# 24GB. For a full fine-tune of the diffusion head, use a 40GB+ GPU (L40S/H100)
# and drop that flag (and/or use LoRA).
ACCELERATOR = truss_config.Accelerator.A10G
GPU_COUNT = 1

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh",
        "./run.sh",
    ],
    environment_variables={
        # Faster downloads for the ~6GB base model.
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        # For gated HF datasets/models, add an `hf_access_token` workspace secret
        # and map it here (the public GR00T base model needs no token):
        #   "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
        enable_legacy_hf_mount=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
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
