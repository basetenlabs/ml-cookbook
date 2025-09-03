from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.2"

training_runtime = definitions.Runtime(
    start_commands = [
        #"/bin/sh -c './run_1node.sh'",
        # "./run_1node.sh"
        "/bin/sh -c 'chmod +x ./run_1node.sh && ./run_1node.sh'"
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"), # The name of the HF Access Token secret in your B10 account
        "HF_HUB_ENABLE_HF_TRANSFER": "true",
        # "HF_HOME": "/root/.cache/user_artifacts/hf_cache",
        # "WORLD_SIZE": "8",  # total gpus
        # "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"), # comment this out if you don't want to use wandb
    },
    cache_config=definitions.CacheConfig(
        enabled=False,
    )
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,  
        count=8,
    ),
    node_count=1,
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)

training_project = definitions.TrainingProject(
    name="MSwift Megatron - qwen3-30b-a3b-base No cache root",
    job=training_job
)
