"""Import a Hugging Face Whisper checkpoint using a CPU-only Training job."""

import os
import re

from truss_train import definitions

repo = os.environ.get("SOURCE_REPO", "openai/whisper-large-v3")
revision = os.environ.get("SOURCE_REVISION")
if not revision:
    revision = (
        "06f233fe06e710322aca913c1bc4249a0d71fce1"
        if repo == "openai/whisper-large-v3"
        else "main"
    )
model_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", repo).strip("-")
job_name = os.environ.get("TRAINING_JOB_NAME")
if not job_name:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Pass --revision with the full HF commit SHA for the default <model-id>-<sha6> "
            "job name, or set --job-name explicitly when importing a branch or tag."
        )
    job_name = f"{model_name[:193]}-{revision[:6]}"

environment_variables = {
    "SOURCE_REPO": repo,
    "SOURCE_REVISION": revision,
    "CHECKPOINT_NAME": os.environ.get("CHECKPOINT_NAME", "checkpoint-0"),
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}
if secret_name := os.environ.get("HF_SECRET_NAME"):
    environment_variables["HF_TOKEN"] = definitions.SecretReference(name=secret_name)

training_project = definitions.TrainingProject(
    name=os.environ.get("TRAINING_PROJECT_NAME", repo),
    job=definitions.TrainingJob(
        name=job_name,
        image=definitions.Image(base_image="python:3.12-slim"),
        compute=definitions.Compute(cpu_count=4, memory="16Gi"),
        runtime=definitions.Runtime(
            start_commands=[
                "pip install --no-cache-dir -r requirements.txt",
                "python3 import_checkpoint.py",
            ],
            environment_variables=environment_variables,
            checkpointing_config=definitions.CheckpointingConfig(enabled=True),
        ),
    ),
)
