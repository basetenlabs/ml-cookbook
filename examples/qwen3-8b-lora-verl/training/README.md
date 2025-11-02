# Qwen3 8B LoRA Training with veRL

This example demonstrates how to train [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B) using [veRL](https://github.com/volcengine/verl) for reinforcement learning from human feedback (RLHF) with LoRA (Low-Rank Adaptation).

## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
2. Accept the terms and request access to the model (if it's gated)

### Update config.py
- Make sure the environment variables in `config.py` match the names of secrets as saved in Baseten secrets.

### Update training configuration
- Review `prepare_dataset.py` to configure your dataset preparation.
- Review `reward_function.py` to configure your reward function.
- Review `run.sh` for training hyperparameters, batch sizes, LoRA settings, and other important parameters.

### Launch run

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.

