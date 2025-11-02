## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
2. Accept the terms and request access to the model (if it's gated)

### Update config.py 
- Make sure the environment variables in `config.py` is updated with the same name of secrets as saved in Baseten secrets.
- In `train.py`, where all the data processing and training code is present, make sure the data processing function `load_and_process_common_accent_dataset` or `load_and_process_dataset` processes your dataset as desired. This file has 2 examples - `load_and_process_common_accent_dataset` or `load_and_process_dataset`
    - This is also where hyper parameters for training are defined. Refer to `run.sh` to change them. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.