from pathlib import Path

from truss_train import definitions
from truss.base.truss_config import AcceleratorSpec

# ---------------------------------------------------------------------------
# Read settings.env (parent dir of training/)
# ---------------------------------------------------------------------------

def _load_settings():
    """Parse settings.env as key=value pairs, ignoring comments and blanks."""
    settings = {}
    settings_path = Path(__file__).parent.parent / "settings.env"
    if settings_path.exists():
        for line in settings_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().split("#")[0].strip()  # strip inline comments
            if key:
                settings[key] = value
    return settings

_settings = _load_settings()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_IMAGE = "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"

training_runtime = definitions.Runtime(
    start_commands=[
        "chmod +x ./run.sh && ./run.sh",
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "MODEL": _settings.get("MODEL", "Qwen/Qwen3-8B"),
        "MODEL_TYPE": _settings.get("MODEL_TYPE", ""),
        "DATASET": _settings.get("DATASET", "winglian/pirate-ultrachat-10k"),
        "EVAL_SPLIT_RATIO": _settings.get("EVAL_SPLIT_RATIO", "0.01"),
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

_accelerator = _settings.get("ACCELERATOR", "H100")
_gpu_count = int(_settings.get("GPU_COUNT", "2"))

training_compute = definitions.Compute(
    accelerator=AcceleratorSpec(accelerator=_accelerator, count=_gpu_count),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name=_settings.get("PROJECT_NAME", "autoresearch-finetune"),
    job=training_job,
)
