"""
Truss Training Configuration

This config is designed to be used with the programmatic training API.
It reads dynamic configuration from runtime_config.json which is written
by the training handler before submission.
"""

import json
from pathlib import Path

from truss.base import truss_config
from truss_train import definitions

# Read runtime configuration (written by training_handler.py)
config_path = Path(__file__).parent / "runtime_config.json"
if config_path.exists():
    runtime_config = json.loads(config_path.read_text())
else:
    # Default values for local development/testing
    runtime_config = {
        "dataset_id": "HuggingFaceH4/Multilingual-Thinking",
        "model_id": "openai/gpt-oss-20b",
        "project_name": "programmatic-training-job",
    }

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = definitions.Runtime(
    start_commands=[
        "pip install 'trl>=0.20.0' 'peft>=0.17.0' 'transformers>=4.55.0'",
        "python3 train.py",
    ],
    environment_variables={
        # Secrets from Baseten account
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # Dynamic configuration passed as environment variables
        "MODEL_ID": runtime_config["model_id"],
        "DATASET_ID": runtime_config["dataset_id"],
    },
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=2,
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name=runtime_config["project_name"],
    job=training_job,
)
