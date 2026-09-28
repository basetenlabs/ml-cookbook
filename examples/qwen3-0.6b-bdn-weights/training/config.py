from truss_train import (
    TrainingProject,
    TrainingJob,
    Image,
    Compute,
    Runtime,
    WeightsSource,
    CacheConfig,
    CheckpointingConfig,
)
from truss.base.truss_config import AcceleratorSpec

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=[
        "pip install transformers datasets accelerate",
        "python train.py",
    ],
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
)

training_compute = Compute(
    accelerator=AcceleratorSpec(accelerator="H100", count=1),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
    weights=[
        # Model weights from Hugging Face, delivered through BDN.
        # Available at mount_location before your training script starts.
        WeightsSource(
            source="hf://Qwen/Qwen3-0.6B",
            mount_location="/app/models/Qwen/Qwen3-0.6B",
        ),
        # Training data from S3, also delivered through BDN.
        # Replace with your own bucket and path.
        # For private buckets, add auth — see:
        # https://docs.baseten.co/development/model/bdn#aws-s3
        #
        # WeightsSource(
        #     source="s3://my-bucket/training-data",
        #     mount_location="/app/data/training-data",
        #     auth={
        #         "auth_method": "CUSTOM_SECRET",
        #         "auth_secret_name": "aws_credentials",
        #     },
        # ),
    ],
)

training_project = TrainingProject(
    name="qwen3-0.6b-bdn-weights",
    job=training_job,
)
