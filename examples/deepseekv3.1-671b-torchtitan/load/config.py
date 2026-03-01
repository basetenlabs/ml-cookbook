from truss.base import truss_config
from truss_train import definitions

BASE_IMAGE = "ghcr.io/pytorch/pytorch-nightly:71739c4-cu12.9.1"
PROJECT_NAME = "DeepSeek-V3.1-LoRA"

ACCELERATOR = truss_config.Accelerator.H200
NGPU = 1
NODE_COUNT = 1

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh && ./run.sh",
    ],
    environment_variables={
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
        require_cache_affinity=True,
    ),
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=ACCELERATOR,
        count=NGPU,
    ),
    node_count=NODE_COUNT,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
    name="LoadDataIntoVolume"
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
