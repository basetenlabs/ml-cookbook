## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (or check your training configuration for the model you're using)
2. Accept the terms and request access to the model (if it's gated)

### Update configuration
- Make sure any configuration files match the names of secrets as saved in Baseten secrets.
- Review all training scripts and hyper parameters to ensure they work for your training use-case.

### Launch run

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.

