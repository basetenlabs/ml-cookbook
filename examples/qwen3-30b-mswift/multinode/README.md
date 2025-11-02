## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
2. Accept the terms and request access to the model (if it's gated)

### Update config.py
- Update `config.py` with environment variables to match your Baseten secrets.
- Configure the number of nodes and compute resources as needed.

### Update run.sh
- If using Weights&Biases to track your run, update `wandb_project` and `wandb_exp_name` accordingly, or comment out these lines.
- Hyper parameters, dataset processing and other important fields are defined in this script. Please go over all fields to make sure they work for your training use-case.

### Launch run

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.

### Accessing checkpoints
To get the location of your checkpoints, run `truss train get_checkpoint_urls --job-id your-job-id`, this downloads a file containing information about your checkpoint storage locations.

