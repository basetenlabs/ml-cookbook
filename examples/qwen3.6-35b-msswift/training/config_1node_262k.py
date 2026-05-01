from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.6-35B-A3B Long Context - ML Cookbook"
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

# 1-node packed 262K — VERIFIED TO OOM. Kept here as a documented dead end so
# readers don't waste compute trying it. Even with recompute_num_layers=1
# (most aggressive: recompute every layer), 262K activations OOM ranks 3 and 7
# during the first iter on a single 8x H200 node. The 128K config already
# peaks at 131 GiB / 141 cap with recompute=2, so 262K's 2x activations don't
# fit even with the recompute lever maxed out. Use config_2node_262k.py for
# the verified PP=2 path.
training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "EXP_TAG": "1n262k",
        "MAX_LENGTH": "262144",
        "TP": "1",
        "PP": "1",
        "EP": "8",
        "CP": "1",
        # Start with the most aggressive recompute since 262K is ~2× the
        # activation footprint of 128K. recompute=1 means recompute every layer.
        "RECOMPUTE_NUM_LAYERS": "1",
        "TRAIN_ITERS": "10",
        "USE_MCORE_GDN": "1",
        "PACKING": "true",
    },
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
    cache_config=definitions.CacheConfig(enabled=True),
)

training_compute = definitions.Compute(
    node_count=1,
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
