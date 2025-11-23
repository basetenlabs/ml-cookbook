## Run instructions

### Update config.yaml 
- Checkpoints are automatically saved to the `output_dir` specified in `config.yaml` (default: `/tmp/training_checkpoints`). The checkpointing configuration in `config.py` is enabled, which downloads these checkpoints to the checkpointing directory for easy access.
- If using Weights&Biases to track your run, update `use_wandb`, `wandb_project` and `wandb_entity` accordingly. 
- Hyper parameters, dataset processing and other important fields are defined in this yaml. Please go over all fields to make sure they work for your training use-case. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.