## Baseten Qwen 80b 2.5 Training

### Prerequisites
https://github.com/basetenlabs/ml-cookbook/tree/main?tab=readme-ov-file#prerequisites

This example uses megatron training
- `run_megatron.sh`

Current files:
- `config.py`: Baseten training job config
- `run_megatron.sh`: Megatron SFT launcher.

### Required Baseten secrets
- `hf_access_token`

### Runtime knobs
Edit variables directly in the script you use:
- `run_megatron.sh`:
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
