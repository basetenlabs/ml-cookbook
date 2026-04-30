"""Debug config: 2 nodes, 8x H200 each, sleep infinity for SSH-driven iteration.

Usage:
    truss train push config_debug_2node.py --team baseten-dogfood --remote baseten
    # then SSH into either node:
    ssh training-job-<job-id>-0.ssh.baseten.co
    ssh training-job-<job-id>-1.ssh.baseten.co

Both nodes share $BT_PROJECT_CACHE_DIR, so deps + model are visible to both.
$BT_GROUP_SIZE=2, $BT_NODE_RANK={0,1}, $BT_LEADER_ADDR points at node 0 — these
are already exported by Baseten before run_debug.sh runs.
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
    node_count=2,
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
