from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.6-35B-A3B Long Context - ML Cookbook"
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # run.sh installs deps and downloads the model in-line on first run
        # (cached after that), then trains. No separate hydrate step needed
        # for single-node sequential experiments.
        "EXP_TAG": "1n128k",
        "MAX_LENGTH": "131072",
        # Verified sweet spot: 1 node, packed 128K, recompute_num_layers=2.
        # ~34 s/iter steady, peak 131 GiB sw. recompute=4 OOMs; recompute=1
        # works at 122 GiB but is ~1.5× slower (50 s/iter). 2-node PP=2 also
        # works (71 s/iter) but is strictly worse than 1-node here.
        "TP": "1",
        "PP": "1",
        "EP": "8",
        "CP": "1",
        "RECOMPUTE_NUM_LAYERS": "2",
        "TRAIN_ITERS": "10",
        # Packing fills max_length so 128K is genuinely exercised.
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
