## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)
2. Accept the terms and request access to the model (if it's gated)

### Update config.yaml 
- If pushing your checkpoints to Huggingface, make sure to update the `hub_model_id` in `config.yaml`, else comment out the lines with `hub_model_id` and `hub_strategy`. 
- If using Weights&Biases to track your run, update `use_wandb`, `wandb_project` and `wandb_entity` accordingly. 
- Hyper parameters, dataset processing and other important fields are defined in this yaml. Please go over all fields to make sure they work for your training use-case. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.