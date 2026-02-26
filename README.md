  <a href="https://www.baseten.co/">
    <img alt="Baseten" src="https://github.com/user-attachments/assets/1e342a9e-56a5-4919-b776-228a5fc1288e">
  </a>
  
<p align="center">
  <strong><a href="https://docs.baseten.co/examples/deploy-your-first-model">Inference docs</a> | <a href="https://docs.baseten.co/training/overview">Training docs</a></strong>
</p>

A curated collection of ready-to-use training recipes for machine learning on Baseten. Whether you’re starting from scratch or fine-tuning an existing model, these recipes provide practical, copy-paste solutions for every stage of your ML pipeline.

### What's inside

- Training recipes - End-to-end examples for training models from scratch
- Fine-tuning workflows - Adapt pre-trained models to your specific use case
- Best practices - Optimized configurations and common patterns

From data preprocessing to checkpointed and trained models, these recipes cover the complete ML lifecycle on Baseten's platform.

### Table of contents

- [Usage](#usage)
  - [Fine-tune GPT OSS 20B with LoRa and trl](#fine-tune-gpt-oss-20b-with-lora-and-trl)
    - [Training](#training)
  - [Fine-tune Qwen3 8B with LoRa and trl](#fine-tune-qwen3-8b-with-lora-and-trl)
    - [Training](#training-1)
  - [Train and deploy an MNIST digit classifier with Pytorch](#train-and-deploy-an-mnist-digit-classifier-with-pytorch)
    - [Training](#training-2)
- [Contributing](#contributing)
- [License](#license)

### Prerequisites

Before getting started, ensure you have the following:

- A Baseten account. [Sign up here](https://baseten.co/signup) if you don't have one.
  - Add any access tokens, API keys (example: Hugging Face access token), passwords to securely access credentials from your models in [secrets](https://app.baseten.co/settings/secrets).
  - This is required to access models on Huggingface that have gated access. More information on setting up Huggingface access tokens can be found [here](https://huggingface.co/docs/hub/en/security-tokens).
- Python 3.8 to 3.11 installed. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) env recommended.

### Run the examples 

#### Install `truss` 
Use the appropriate command for your package manager
```bash
# pip
pip install -U truss
# uv
uv add truss && uv sync --upgrade-package truss
```

#### Create the workspace for your training project

```bash
# for any example (replace with the specific example name)
truss train init --examples <example-name> && cd <example-name>
```

#### Kick off the job

Make sure you've plugged in proper secrets (e.g. Hugging Face token) via Baseten Secrets and Environment Variables, and kick off your job

```bash
truss train push config.py
```

For more details, take a look at the [docs](https://docs.baseten.co/training/overview)

### If you'd like to fire off jobs from within this repository directly, you can clone the respository and navigagte to the approriate workspaces.

```bash
git clone https://github.com/basetenlabs/ml-cookbook.git
```

## Usage

### Fine-tune GPT OSS 20B with LoRa and [trl](https://github.com/huggingface/trl)

If using a model with gated access, make sure you have access to the model on HuggingFace and your API token uploaded to your [secrets](https://app.baseten.co/settings/secrets). This example requires an [HF access token](https://huggingface.co/docs/hub/en/security-tokens).

#### Training

`examples/oss-gpt-20b-lora/training/train.py` contains all training code.

`examples/oss-gpt-20b-lora/training/config.py` will be the entry point to start training, where you can [define your training configuration](https://docs.baseten.co/training/getting-started#step-1%3A-define-your-training-configuration). This also includes the start commands to launch your training job. Make sure these commands also include any file permission changes to make shell scripts run. We do not change any file system permissions.

Make sure to update `hf_access_token` in `config.py` with the same name for this access token saved in your [secrets](https://app.baseten.co/settings/secrets). In this example, we will be writing trained checkpoints directly to Huggingface, the Hub IDs for models and datasets are configured in `examples/oss-gpt-20b-lora/training/run.sh`. Update `run.sh` with a repo you have access to write to.

```bash
cd examples/oss-gpt-20b-lora/training
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job:

```
✨ Training job successfully created!
🪵 View logs for your job via `truss train logs --job-id e3m512w [--tail]`
🔍 View metrics for your job via `truss train metrics --job-id e3m512w`
```

Keep the Job ID handy, as you’ll use it for [managing and monitoring your job](https://docs.baseten.co/training/management).

Alternatively, you can view all your training jobs at (https://app.baseten.co/training/)[https://app.baseten.co/training/].

- As checkpoints are generated, you can access them on Huggingface at the same location defined in `run.sh`.

### Fine-tune Qwen3 8B with LoRa and [trl](https://github.com/huggingface/trl)

If using a model with gated access, make sure you have access to the model on HuggingFace and your API token uploaded to your [secrets](https://app.baseten.co/settings/secrets).

#### Training

`examples/qwen3-8b-lora-dpo-trl/training/train.py` contains the training code.

`examples/qwen3-8b-lora-dpo-trl/training/config.py` will be the entry point to start training, where you can [define your training configuration](https://docs.baseten.co/training/getting-started#step-1%3A-define-your-training-configuration). This also includes the start commands to launch your training job. Make sure these commands also include any file permission changes to make shell scripts run. We do not change any file system permissions.

```bash
cd examples/qwen3-8b-lora-dpo-trl/training
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job:

```
✨ Training job successfully created!
🪵 View logs for your job via `truss train logs --job-id e3m512w [--tail]`
🔍 View metrics for your job via `truss train metrics --job-id e3m512w`
```

Alternatively, you can view all your training jobs at (https://app.baseten.co/training/)[https://app.baseten.co/training/].

In this example, since checkpointing is enabled in `config.py`, checkpoints are stored in cloud storage and can be accessed with

```
truss train get_checkpoint_urls --job-id $JOB_ID
```

### Train and deploy an MNIST digit classifier with Pytorch

#### Training

`examples/mnist-single-gpu/training/train_mnist.py` contains the a Pytorch example of an MNIST classifier with CNNs.

`examples/mnist-single-gpu/training/config.py` will be the entry point to start training, where you can [define your training configuration](https://docs.baseten.co/training/getting-started#step-1%3A-define-your-training-configuration). This also includes the start commands to launch your training job. Make sure these commands also include any file permission changes to make shell scripts run. We do not change any file system permissions.

```bash
cd examples/mnist-single-gpu/training
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job:

```
✨ Training job successfully created!
🪵 View logs for your job via `truss train logs --job-id e3m512w [--tail]`
🔍 View metrics for your job via `truss train metrics --job-id e3m512w`
```

Keep the Job ID handy, as you’ll use it for [managing and monitoring your job](https://docs.baseten.co/training/management).

In this example, since checkpointing is enabled in `config.py`, checkpoints are stored in cloud storage and can be accessed with

```
truss train get_checkpoint_urls --job-id $JOB_ID
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

[MIT License](LICENSE)
