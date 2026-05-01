from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.6-35B-A3B Long Context - ML Cookbook"
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

# 2-node packed 262K — STILL OOMs as of last test (PP=2 + recompute=1).
# The OOM is on node 1's pipeline stage (ranks 8-15), driven by the LM head
# logits tensor (seq × vocab = 131072 × 248320 × 2 bytes ≈ 65 GB unsplit) +
# the 1F1B microbatch activation buffer that backs up on the trailing stage.
#
# Open follow-up to try: TP=2 + PP=2 + EP=4. TP=2 splits the LM head logits
# along the vocab dim, halving the dominant tensor on stage 2. EP drops from
# 8→4 because TP=2 × PP=2 leaves DP=4 with 16 GPUs (each rank then holds 64
# experts vs 32 — adds ~400 MB/rank, trivial vs the ~33 GB saved). Requires
# megatron-core ≥ 0.16 (we have 0.16.1) which lifted the old
# num_query_groups=2 ⇒ TP≤2 cap.
#
# Until that's verified, this config is documented but unconfirmed.
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
