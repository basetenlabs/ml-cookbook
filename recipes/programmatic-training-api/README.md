# Programmatic Training API Integration

This recipe demonstrates how to programmatically kick off training jobs on Baseten. This is useful when you want to:

- Build an API endpoint that receives training requests (e.g. pearametrized on data or a model)
- Dynamically configure training jobs based on user input
- Integrate training into your application workflow

## Overview

The pattern works by:
1. Maintaining a "template" training directory with your training code
2. Copying the template to a temporary directory for each request
3. Injecting dynamic configuration (dataset_id, model_id, etc.)
4. Using `truss_train.public_api.push()` to submit the job

## Directory Structure

```
programmatic-training-api/
├── README.md
├── requirements.txt          # Dependencies for the handler
├── training_handler.py       # Main API handler module
└── training_template/        # Template training files
    ├── config.py             # Truss training configuration
    └── train.py              # Training script
```

## Usage

### Basic Example

```python
from training_handler import handle_training_request

# Kick off a training job with custom dataset and model
result = handle_training_request(
    dataset_id="HuggingFaceH4/Multilingual-Thinking",
    model_id="openai/gpt-oss-20b",
    project_name="my-custom-training-job"
)

print(f"Training Project: {result['training_project']}")
print(f"Training Job: {result['training_job']}")
```

### Integration with a Web Framework (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from training_handler import handle_training_request

app = FastAPI()

class TrainingRequest(BaseModel):
    dataset_id: str
    model_id: str
    project_name: str | None = None

@app.post("/train")
async def create_training_job(request: TrainingRequest):
    result = handle_training_request(
        dataset_id=request.dataset_id,
        model_id=request.model_id,
        project_name=request.project_name,
    )
    return {
        "training_project_id": result["training_project"].id,
        "training_job_id": result["training_job"].id,
    }
```

## Configuration

### Environment Variables

The training template uses environment variables to pass dynamic configuration:

| Variable | Description |
|----------|-------------|
| `MODEL_ID` | HuggingFace model ID to fine-tune |
| `DATASET_ID` | HuggingFace dataset ID for training |
| `PROJECT_NAME` | Name for the training project |

### Secrets

Make sure you have the following secrets configured in your Baseten account:

- `hf_access_token` - HuggingFace access token for model/dataset access

## Customization

### Modifying Training Parameters

Edit `training_template/train.py` to adjust:
- LoRA configuration (rank, alpha, target modules)
- Training hyperparameters (learning rate, batch size, epochs)
- Model quantization settings

### Changing Compute Resources

Edit `training_template/config.py` to adjust:
- GPU type and count
- Node count for multi-node training
- Base Docker image

### Adding More Dynamic Parameters

1. Add the parameter to the `runtime_config` dict in `training_handler.py`
2. The config is passed via the `RUNTIME_CONFIG` environment variable
3. Use `os.environ.get()` in `training_template/train.py` to read it

## API Reference

### `handle_training_request()`

```python
def handle_training_request(
    dataset_id: str,
    model_id: str,
    project_name: str | None = None,
    training_template_dir: Path | None = None,
) -> dict:
    """
    Handle a training request by creating and submitting a training job.

    Args:
        dataset_id: HuggingFace dataset ID for training data
        model_id: HuggingFace model ID to fine-tune
        project_name: Optional name for the training project
        training_template_dir: Optional path to custom training template

    Returns:
        dict: Contains 'training_project' and 'training_job' objects
    """
```

## Response Format

The `push()` function returns:

```python
{
    "training_project": TrainingProject,  # Project metadata
    "training_job": TrainingJob,          # Job metadata with status
}
```

For detailed field definitions, see the [Training API docs](https://docs.baseten.co/reference/training-api/get-training-job).

## After Pushing
* Request [logs](https://docs.baseten.co/reference/training-api/get-training-job-logs) and [metrics](https://docs.baseten.co/reference/training-api/get-training-job-metrics) after pushing the job to assess progress of the training job via API.
* Check the [training job's status](https://docs.baseten.co/reference/training-api/get-training-job).
* [List the checkpoints](https://docs.baseten.co/reference/training-api/get-training-job-checkpoints) associated with the project
