"""
Programmatic Training API Integration Handler

This module demonstrates how to programmatically kick off training jobs
on Baseten by dynamically configuring and submitting training projects.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from truss_train.public_api import push


def handle_training_request(
    dataset_id: str,
    model_id: str,
    project_name: Optional[str] = None,
    training_template_dir: Optional[Path] = None,
) -> dict:
    """
    Handle a training request by creating and submitting a training job.

    This function:
    1. Copies the training template to a temporary directory
    2. Writes runtime configuration with the provided parameters
    3. Submits the training job using the truss_train API

    Args:
        dataset_id: HuggingFace dataset ID for training data
        model_id: HuggingFace model ID to fine-tune
        project_name: Optional name for the training project.
                     Defaults to "training-{model_id}"
        training_template_dir: Optional path to custom training template.
                              Defaults to ./training_template

    Returns:
        dict: A dictionary containing:
            {
                "training_project": TrainingProject,
                "training_job": TrainingJob,
            }

    Example:
        >>> result = handle_training_request(
        ...     dataset_id="HuggingFaceH4/Multilingual-Thinking",
        ...     model_id="openai/gpt-oss-20b",
        ... )
        >>> print(result["training_job"].id)
    """
    # Resolve paths
    if training_template_dir is None:
        training_template_dir = Path(__file__).parent / "training_template"

    if project_name is None:
        # Create a safe project name from model_id
        project_name = f"training-{model_id.replace('/', '-')}"

    # Create temporary directory for this training job
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Copy training template to temp directory
        shutil.copytree(training_template_dir, tmp_path, dirs_exist_ok=True)

        # Write runtime configuration
        runtime_config = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "project_name": project_name,
        }

        config_path = tmp_path / "runtime_config.json"
        config_path.write_text(json.dumps(runtime_config, indent=2))

        # Submit training job
        config_py_path = tmp_path / "config.py"
        result = push(config=config_py_path, remote="baseten")

        return result


def handle_training_request_with_custom_params(
    dataset_id: str,
    model_id: str,
    project_name: Optional[str] = None,
    training_template_dir: Optional[Path] = None,
    # Training hyperparameters
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    # LoRA parameters
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> dict:
    """
    Handle a training request with customizable training parameters.

    This extended version allows passing training hyperparameters
    that will be injected into the training configuration.

    Args:
        dataset_id: HuggingFace dataset ID for training data
        model_id: HuggingFace model ID to fine-tune
        project_name: Optional name for the training project
        training_template_dir: Optional path to custom training template
        learning_rate: Learning rate for training (default: 2e-4)
        num_epochs: Number of training epochs (default: 1)
        batch_size: Per-device batch size (default: 4)
        max_seq_length: Maximum sequence length (default: 2048)
        lora_r: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling factor (default: 16)

    Returns:
        dict: Contains 'training_project' and 'training_job' objects
    """
    if training_template_dir is None:
        training_template_dir = Path(__file__).parent / "training_template"

    if project_name is None:
        project_name = f"training-{model_id.replace('/', '-')}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copytree(training_template_dir, tmp_path, dirs_exist_ok=True)

        # Write extended runtime configuration
        runtime_config = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "project_name": project_name,
            "training_params": {
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "max_seq_length": max_seq_length,
            },
            "lora_params": {
                "r": lora_r,
                "alpha": lora_alpha,
            },
        }

        config_path = tmp_path / "runtime_config.json"
        config_path.write_text(json.dumps(runtime_config, indent=2))

        config_py_path = tmp_path / "config.py"
        result = push(config=config_py_path, remote="baseten")

        return result


# Example usage
if __name__ == "__main__":
    # Basic usage
    result = handle_training_request(
        dataset_id="HuggingFaceH4/Multilingual-Thinking",
        model_id="openai/gpt-oss-20b",
        project_name="my-training-job",
    )

    print(f"Training Project: {result['training_project']}")
    print(f"Training Job: {result['training_job']}")
