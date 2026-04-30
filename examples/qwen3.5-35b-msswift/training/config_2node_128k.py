from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.5-35B-A3B Long Context - ML Cookbook"
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "EXP_TAG": "2n128k",
        "MAX_LENGTH": "131072",
        # Verified working with packing=true (true 128K sequences, not LongAlign's
        # ~64K-max samples). PP=2 splits 40 layers into 2 stages of 20, halving
        # per-stage activations under recompute. EP=8 still shards 256 experts
        # cleanly. CP unavailable: mcore-bridge 1.2.x asserts "Gated delta net
        # does not support context parallel for now" — needs megatron-core from
        # git main, which we don't install. PP is the practical multi-node lever.
        "TP": "1",
        "PP": "2",
        "EP": "8",
        "CP": "1",
        "RECOMPUTE_NUM_LAYERS": "4",
        "TRAIN_ITERS": "10",
        # USE_MCORE_GDN=1 is required for packing on the GDN sublayers.
        "USE_MCORE_GDN": "1",
        # Packing fills max_length so the activation cost actually matches the
        # configured ceiling (not the dataset's longest unpacked sample).
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
