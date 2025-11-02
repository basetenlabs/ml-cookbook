# Orpheus + Transformers library
This example demonstrates how to train [Orpheus](https://github.com/canopyai/Orpheus-TTS) on a specific voice dataset by training a [LoRA module](https://www.ibm.com/think/topics/lora). This allows you to customize the powerful Text-To-Speech (TTS) model to use accents and tones that fit your use case best.

## Run instructions

### Set up secrets
Before running, you need to configure your secrets in Baseten:
1. Go to **Settings** → **Secrets**
2. Set `hf_access_token` - Your Hugging Face access token (required for downloading models)
3. Optionally set `wandb_api_key` - Your Weights & Biases API key (if using W&B for experiment tracking)

### Ungate models on Hugging Face
If you're using a gated model, you need to ungate it on Hugging Face:
1. Go to the model page: [unsloth/orpheus-3b-0.1-ft](https://huggingface.co/unsloth/orpheus-3b-0.1-ft) (or check your `train.py` script for the model you're using)
2. Accept the terms and request access to the model (if it's gated)

### Update config.py 
- Make sure the environment variables in `config.py` is updated with the same name of secrets as saved in Baseten secrets.
- `train.py` has the data processing calls, training code and hyper parameters defined. Update datasets, flags etc here or pass them as arguments through `run.sh` to change them. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.
