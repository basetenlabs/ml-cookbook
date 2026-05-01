from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.6-35B-A3B Long Context - ML Cookbook"
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

# 2-node packed 262K. 1-node packed 262K OOMs even with recompute_num_layers=1
# (262K activations are ~2× 128K's, and 128K already filled most of an H200).
# PP=2 splits the 40 layers into 2 stages of 20, halving per-stage activation
# memory; combined with recompute=2 this is the configuration we expect to fit.
training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "EXP_TAG": "2n262k",
        "MAX_LENGTH": "262144",
        "TP": "1",
        "PP": "2",
        "EP": "8",
        "CP": "1",
        "RECOMPUTE_NUM_LAYERS": "1",
        "TRAIN_ITERS": "10",
        "USE_MCORE_GDN": "1",
        "PACKING": "true",
    },
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
    cache_config=definitions.CacheConfig(enabled=True),
)

training_compute = definitions.Compute(
    node_count=2,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H200, count=8
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
