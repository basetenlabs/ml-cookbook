from truss.base.truss_config import AcceleratorSpec
from truss_train import (
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    Runtime,
    TrainingJob,
    TrainingProject,
)

training_job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=4)),
    runtime=Runtime(
        start_commands=["bash run.sh"],
        environment_variables={"ROWS": "2000"},
        cache_config=CacheConfig(enabled=True),
        checkpointing_config=CheckpointingConfig(enabled=True),
    ),
)

training_project = TrainingProject(name="elephant-1-sft", job=training_job)
