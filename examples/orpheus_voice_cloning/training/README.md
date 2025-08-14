# Baseten Finetune Demo

This is a demo of how to finetune Orpheus for voice cloning using Baseten with Unsloth.

## Setup

```bash
pip install -U truss # should be >= 0.9.84
```

## Define your training config

You can see the example in `config.py`. You can add local artifacts to be uploaded to the container, like `train.py`, `run.sh`, `deepspeed_config.json`, etc. These artifacts will be copied into the container at the root of the base image's workdir.

## Prerequisites

Set your hf_access_token and wandb_access_token in your Baseten dashboard. 

## Run Training

```bash
truss train push config.py
```
After pushing the job, you'll see the project id and job id output to the console. You can use these to monitor the job and stop the job. 

### Monitor
To view the logs for the most recent job, you can run 
```bash
truss train logs --tail # logs 
truss train metrics # metrics
```
If you have more than one job running, you can specify the `job-id`. 

To view your jobs and projects, you can run 
```bash
truss train view # view all projects and active jobs
truss train view --project-id <project-id> # view all jobs for a project
```

In case of problems, additional information is available in the [Model
Training Debug
dashboard](https://grafana.baseten.co/d/cen5oedvwxds0a/training-job-debug-dash). Use
the training job ID to filter. Use the workplane information from
Billfred to filter to the proper cluster for logs and metrics.

### Stop
```bash
# stops the active job if there is only one. Will ask for confirmation.
truss train stop
# stop a specific job
truss train stop --job-id <job-id>
# stop all active jobs. Will ask for confirmation.
truss train stop --all
```

## Run Prediction

After training is complete, you can run prediction on the model.
```bash
truss train deploy_checkpoints --job-id <job-id>
```

## Concepts

### TrainingProject

A project is a collection of jobs. Caches are shared across jobs in a specific project. This means that if you have a large base model or dataset, you can cache these resources so that they are only downloaded once. See [Caches](#caches) for more details.

### TrainingJob

A TrainingJob is a single job. It is defined by the compute, the base image, and the runtime environment, as specified in the pythonic config file.

A training job has a lifecycle that is managed by the Baseten platform. The lifecycle of a training job is as follows:

1. `TRAINING_JOB_CREATED` - The job has been created on the Baseten platform
2. `TRAINING_JOB_DEPLOYING` - The job is being deployed 
3. `TRAINING_JOB_RUNNING` - The job is running
4. `TRAINING_JOB_COMPLETED` - The job completed successfully
5. `TRAINING_JOB_FAILED` - The job unexpectedly failed to complete
6. `TRAINING_JOB_STOPPED` - Manually stopped by the user


### Caches

Baseten allows you to create a "cache" resource across `TrainingJob`s in a `TrainingProject`. This means that if you have a large base model or dataset, you can cache these resources so that they are only downloaded once. This is defined in the `Runtime` configuration by setting `enable_cache=True`. The default cache size is 1000GB. If this is not enough, reach out to Baseten support.

The Cache is a read/write cache to allow you the freedom to update the cache as you see fit. However, this means that you should be careful about concurrent write access (e.g. creating multiple jobs that write to the cache, or writing to the cache in a multi-threaded process).

The cache is mounted at two points: `/root/.cache/huggingface/hub` and `$BT_RW_CACHE_DIR`, which is currently set to `/root/.cache/user_artifacts`. This is so that huggingface resources are automatically cached and any other resources can be stored independently. Together, these directories share the storage resources that are allocated to the cache.

When hydrating the cache, it is important to do this in a single threaded way to avoid concurrent write access. This is particularly important if your training script is multi-threaded. If so, first download the resources in a separate process. 
```bash
huggingface-cli download google/gemma-3-27b-it --repo-type=model
accelerate launch --config_file config.yml --num_processes 8 train.py
```

To clear the cache, you can `push` a job that deletes the contents of the cache. 
```python
# config.py
training_job = definitions.TrainingJob(
    ...
    runtime=definitions.Runtime(
        start_commands=[
            "rm -rf $BT_RW_CACHE_DIR/*",
            "rm -rf /root/.cache/huggingface/*",
            "ls -la $BT_RW_CACHE_DIR",
            "ls -la /root/.cache/huggingface",
        ],
        enable_cache=True,
    ),
    ...
)
```
