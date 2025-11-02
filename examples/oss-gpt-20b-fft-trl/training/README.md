# GPT-OSS 20B Full Fine-Tuning with TRL

This example demonstrates how to perform full fine-tuning of [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) using [TRL](https://github.com/huggingface/trl) with DeepSpeed ZeRO-3.

## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
2. Accept the terms and request access to the model (if it's gated)

### Update config.py
- Make sure the environment variables in `config.py` match the names of secrets as saved in Baseten secrets.

### Update training configuration
- Review `sft_full.yaml` for training hyperparameters, dataset configuration, and other important settings.
- Review `zero3.yaml` for DeepSpeed ZeRO-3 configuration.
- Update `run.sh` to modify training parameters if needed.

### Launch run

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.

