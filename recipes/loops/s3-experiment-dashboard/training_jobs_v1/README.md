# Training Jobs v1 producer

This adapter publishes Training Jobs v1 metrics to the same S3 contract as the Loops dashboard recipe. It uses the job ID as the run ID. It also records the Baseten project and job metadata automatically.

Training Jobs v1 injects these values into every job:

- `BT_TRAINING_JOB_ID`
- `BT_TRAINING_JOB_NAME`
- `BT_TRAINING_PROJECT_ID`
- `BT_TRAINING_PROJECT_NAME`
- `BT_NODE_RANK`

The helper also checks `RANK` and `LOCAL_RANK`. Only the primary process writes and uploads files.

## 1. Add S3 credentials to the job

Create Baseten workspace secrets for a customer-managed S3 bucket. Grant write access only to the experiment prefix.

Add these values to the `Runtime` in the Training Jobs v1 `config.py`:

```python
from truss_train import definitions

training_runtime = definitions.Runtime(
    start_commands=[
        "pip install -r requirements.txt",
        "python train.py",
    ],
    environment_variables={
        "EXPERIMENT_S3_URI": "s3://my-bucket/my-team/experiments",
        "AWS_ACCESS_KEY_ID": definitions.SecretReference(
            name="experiment_store_aws_access_key_id"
        ),
        "AWS_SECRET_ACCESS_KEY": definitions.SecretReference(
            name="experiment_store_aws_secret_access_key"
        ),
        "AWS_DEFAULT_REGION": "us-east-1",
    },
)
```

Add `boto3>=1.34` to the training project's `requirements.txt`. Copy `s3_experiment_logger.py` next to the training script.

Do not write these files into the Baseten checkpoint directory. The checkpoint service manages that directory as model output. This adapter writes to `/tmp/s3-experiment-dashboard/<job_id>` and uploads to the customer bucket.

## 2. Log training metrics

Use the helper as a context manager around the training loop:

```python
from s3_experiment_logger import S3ExperimentLogger

hyperparameters = {
    "model": "Qwen/Qwen3-8B",
    "learning_rate": 3e-5,
    "batch_size": 32,
    "max_steps": 1_000,
}

with S3ExperimentLogger(hyperparameters, sync_every_seconds=60) as experiment:
    for step in range(1, 1_001):
        train_loss = train_one_step()
        experiment.log(
            step,
            train_loss=train_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
```

Call `experiment.log(...)` after evaluation to add `eval_loss`, accuracy, reward, or other numeric values. The dashboard charts every numeric key.

The helper uploads all three files at startup, every 60 seconds while metrics arrive, and at normal process exit. Upload failures produce training logs and do not stop training by default. Pass `strict_uploads=True` if the experiment record must be complete.

## 3. Open the dashboard

On a local machine with read access to the same bucket:

```bash
cd recipes/loops/s3-experiment-dashboard
S3_URI=s3://my-bucket/my-team/experiments ./sync.sh
S3_URI=s3://my-bucket/my-team/experiments python3 build_manifests.py
python3 dashboard.py
```

The compare page identifies these runs with `source: baseten-training-jobs-v1`. Each row uses the Baseten training job ID as its run ID.

## Test the adapter

The test uses a fake S3 client. It does not need AWS credentials.

```bash
python3 -m unittest training_jobs_v1/test_s3_experiment_logger.py
```
