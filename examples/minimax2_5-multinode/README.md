## Baseten MiniMax 2.5 Training (Multi-Node, ms-swift)

This example uses megatron training
- `run_msswift.sh` (Megatron path with TP/EP and FP8 training flags).

Current files:
- `config.py`: Baseten training job config (2 nodes x 8 GPUs, H200 in current config).
- `run_msswift.sh`: Megatron SFT launcher.

### Required Baseten secrets
- `hf_access_token`
- `wandb_api_key` (optional if reporting is disabled)

### Runtime knobs
Edit variables directly in the script you use:
- `run_msswift.sh`:
  - `MODEL_ID`, `DATASET_ID`, `DATASET_SPLIT`
  - `LORA_RANK`, `LORA_ALPHA`
  - `TENSOR_PARALLEL_SIZE`, `PIPELINE_PARALLEL_SIZE`, `CONTEXT_PARALLEL_SIZE`, `EXPERT_PARALLEL_SIZE`
  - `MICRO_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `MAX_LENGTH`
  - `LR_DECAY_STYLE`, `MIN_LR`, `SAVE_INTERVAL`, `LOG_INTERVAL`, `SAVE_FULL_MODEL`

### Launch
`config.py` runs whichever script is in `training_runtime.start_commands`.

Run:
```bash
truss train push config.py
```
