"""Debug config: 1 node, 8x H200, sleep infinity so we can SSH in and iterate.

Usage:
    truss train push config_debug.py --team baseten-dogfood --remote baseten
    # then SSH:
    ssh training-job-<job-id>-0.ssh.baseten.co

Once inside the pod, the deps and model snapshot are already in
$BT_PROJECT_CACHE_DIR. Source the helper:
    export PYTHONPATH=$BT_PROJECT_CACHE_DIR/qwen3_5_packages:$PYTHONPATH
    export HF_HOME=$BT_PROJECT_CACHE_DIR/huggingface
    export USE_MCORE_GDN=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Then iterate on the megatron sft command directly.
"""
from truss_train import definitions
from truss.base import truss_config

project_name = "Qwen3.5-35B-A3B Long Context - ML Cookbook"
BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

training_runtime = definitions.Runtime(
    start_commands=["chmod +x ./run_debug.sh && ./run_debug.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
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
    interactive_session=definitions.InteractiveSession(
        trigger=definitions.InteractiveSessionTrigger.ON_DEMAND,
        session_provider=definitions.InteractiveSessionProvider.SSH,
    ),
)

first_project_with_job = definitions.TrainingProject(
    name=project_name, job=my_training_job
)
