## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Update config.py
- Make sure the environment variables in `config.py` match the names of secrets as saved in Baseten secrets.
- Review the training script `train_mnist.py` and hyper parameters to ensure they work for your training use-case.

### Launch run

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.

