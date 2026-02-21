## Baseten MiniMax 2.5 Training (Multi-Node, DeepSpeed + LoRA)

This repo now uses a multinode `torchrun` launch with Transformers + PEFT + DeepSpeed.

### What changed
- `config.py`: Baseten job config with 2 nodes x 8xH100.
- `run.sh`: installs dependencies, sets debug envs, and launches multinode `torchrun`.
- `train_ds.py`: training script that loads MiniMax M2.5, applies LoRA, and trains with DeepSpeed.
- `ds_config.json`: DeepSpeed ZeRO Stage 3 config used by `TrainingArguments`.

### Required Baseten secrets
- `hf_access_token`
- `wandb_api_key` (optional if you remove/reporting integrations)

### Runtime knobs
Edit variables directly in `run.sh`:
- `MODEL_ID` (default `MiniMaxAI/MiniMax-M2.5`)
- `DATASET_ID`, `DATASET_SPLIT`
- `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`
- `MICRO_BATCH_SIZE`, `GRAD_ACC_STEPS`, `LEARNING_RATE`
- `NUM_EPOCHS` or `MAX_STEPS`
- `MAX_LENGTH`, `SAVE_STEPS`, `LOGGING_STEPS`

### Launch
```bash
truss train push config.py
```

Baseten will print a job id for logs and checkpoint tracking.

### Optional: ms-swift Variant
This repo also includes an ms-swift/Megatron attempt path:
- `/Users/ervinwang/Documents/trussclone/mm_2_5/config_msswift.py`
- `/Users/ervinwang/Documents/trussclone/mm_2_5/run_msswift.sh`

Launch it with:
```bash
truss train push config_msswift.py
```
