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

# H100 (80GB): a full fine-tune of GR00T's projector + diffusion action head fits
# comfortably, and the box's large host RAM handles the full 3B-model save.
# (A10G trains GR00T fine, but its 16GB host RAM OOMs at the full save — use LoRA
# there; see run.sh.)
ACCELERATOR = truss_config.Accelerator.H100
GPU_COUNT = 1

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh",
        "./run.sh",
    ],
    environment_variables={
        # For gated HF datasets/models, add an `hf_access_token` workspace secret
        # and map it here (the public GR00T base model needs no token):
        #   "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
        enable_legacy_hf_mount=True,
        # Don't pin the HF cache to one GPU type's nodes — lets the same project
        # run on A10G or H100 without a cache-affinity conflict.
        require_cache_affinity=False,
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
