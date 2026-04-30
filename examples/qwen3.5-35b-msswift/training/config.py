from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.5-35B-A3B Long Context - ML Cookbook"

# Same image used by the qwen3-80b-msswift example. run.sh upgrades ms-swift
# (3.10.3 → >=4.1.0) and transformers (→ 5.2.*) at runtime, since Qwen3.5
# requires those for model registration + the Megatron-native GatedDeltaNet.
# It also installs flash-linear-attention + causal-conv1d, which the GDN
# sublayers depend on.
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh && ./run.sh",
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # Set to "1" for the first run to install deps and snapshot the model
        # into the project cache. Flip to "0" once cache is hydrated.
        "HYDRATE_ONLY": "1",
        "EXP_TAG": "hydrate",
    },
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H200,
        count=8,
    ),
)

my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

first_project_with_job = definitions.TrainingProject(
    name=project_name, job=my_training_job
)
