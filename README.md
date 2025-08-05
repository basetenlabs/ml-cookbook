<div align="center">

  <a href="https://www.baseten.co/"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/618081de-e4ff-425f-aac7-14e7ca29c03b">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/618081de-e4ff-425f-aac7-14e7ca29c03b">
    <img alt="baseten logo" src="" height="110" style="max-width: 100%;">
  </picture></a>
  
<a href="https://docs.baseten.co/examples/deploy-your-first-model"><img src="https://github.com/user-attachments/assets/043ac2bc-60cf-485e-8d9a-467bfa69a5e0" width="154"></a>
<a href="https://docs.baseten.co/training/overview"><img src="https://github.com/user-attachments/assets/79bee104-2a0a-45f0-ac4a-7a832a1d5a2a" width="165"></a>

### From training to serving - recipes for every ML appetite!

</div>

# ML Training & Deployment Workflows
A collection of ready-to-use training recipes, fine-tuning examples, and deployment workflows for machine learning on Baseten. Whether you're training from scratch, fine-tuning existing models, or deploying for inference, these recipes provide practical, copy-paste solutions for your ML pipeline.

### What's Inside

- Training recipes - End-to-end examples for training models from scratch
- Fine-tuning workflows - Adapt pre-trained models to your specific use case
- Deployment guides - Get your trained models serving predictions in production
- Best practices - Optimized configurations and common patterns

From data preprocessing to production deployment, these recipes cover the complete ML lifecycle on Baseten's platform.

# Table of Contents

- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Recipes](#recipes)
- [Contributing](#contributing)
- [License](#license)


## Getting Started

### Prerequisites

Before getting started, ensure you have the following:

- A Baseten account. [Sign up here](https://baseten.co/signup) if you don't have one.
    - Add any access tokens, API keys (Example: Huggingface access token, Weights&Biases access token), passwords to securely access credentials from your models in [secrets](https://app.baseten.co/settings/secrets).
- Python 3.8 to 3.11 installed. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) env recommended.
- Install [Truss](https://github.com/basetenlabs/truss), Baseten's open-source model packaging tool to configure and containerize model code.
    - ``` pip install --upgrade truss```


### Clone this repository

```bash
git clone https://github.com/basetenlabs/ml-cookbook.git
```

## Usage

### Fine-tune GPT OSS 20B with LoRa and [trl](https://github.com/huggingface/trl)

If using a model with gated access, make sure you have access to the model on HuggingFace and your API token uploaded to your [secrets](https://app.baseten.co/settings/secrets).

#### Training

`examples/oss-gpt-20b-lora/training/train.py` contains all training code. 

`examples/oss-gpt-20b-lora/training/config.py` will be the entry point to start training, where you can [define your training configuration](https://docs.baseten.co/training/getting-started#step-1%3A-define-your-training-configuration). This also includes the start commands to launch your training job. Make sure these commands also include any file permission changes to make shell scripts run. We do not change any file system permissions. 

Make sure to update `hf_access_token` in `config.py` with the same name for this access token saved in your [secrets](https://app.baseten.co/settings/secrets). 

```bash
cd examples/oss-gpt-20b-lora/training
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job:

```json
✨ Training job successfully created!
🪵 View logs for your job via `truss train logs --job-id e3m512w [--tail]`
🔍 View metrics for your job via `truss train metrics --job-id e3m512w`
```

Keep the Job ID handy, as you’ll use it for [managing and monitoring your job](https://docs.baseten.co/training/management).

In this example, since checkpointing is enabled in `config.py`, checkpoints are stored in cloud storage and can be accessed with 
```
truss train get_checkpoint_urls --job-id $JOB_ID
```


#### Inference

...



### Fine-tune Llama 3.1 8b Instruct with LoRa and [Unsloth](https://github.com/unslothai/unsloth/tree/main)

If using a model with gated access, make sure you have access to the model on HuggingFace and your API token uploaded to your [secrets](https://app.baseten.co/settings/secrets).

#### Training

`examples/llama-finetune-8b-lora/training/train.py` contains the training code. 

`examples/llama-finetune-8b-lora/training/config.py` will be the entry point to start training, where you can [define your training configuration](https://docs.baseten.co/training/getting-started#step-1%3A-define-your-training-configuration). This also includes the start commands to launch your training job. Make sure these commands also include any file permission changes to make shell scripts run. We do not change any file system permissions. 

```bash
cd examples/llama-finetune-8b-lora/training
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job:

```json
✨ Training job successfully created!
🪵 View logs for your job via `truss train logs --job-id e3m512w [--tail]`
🔍 View metrics for your job via `truss train metrics --job-id e3m512w`
```

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

```json
✨ Training job successfully created!
🪵 View logs for your job via `truss train logs --job-id e3m512w [--tail]`
🔍 View metrics for your job via `truss train metrics --job-id e3m512w`
```

Keep the Job ID handy, as you’ll use it for [managing and monitoring your job](https://docs.baseten.co/training/management).

In this example, since checkpointing is enabled in `config.py`, checkpoints are stored in cloud storage and can be accessed with 
```
truss train get_checkpoint_urls --job-id $JOB_ID
```

#### Inference

Once you choose a checkpoint to deploy, copy that URL over to ...

To run inference:
```bash
cd examples/mnist-single-gpu/inference
truss push
```


### Fine-tune Llama 3.1 8b Instruct with LoRa and [Unsloth](https://github.com/unslothai/unsloth/tree/main)

- If using a model with gated access, make sure you have access to the model on HuggingFace and your API token uploaded to your [secrets](https://app.baseten.co/settings/secrets).

To save checkpoints to HF ... 

```bash
cd examples/llama_finetune_multi_gpu
truss train push config.py
```


## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

[MIT License](LICENSE)