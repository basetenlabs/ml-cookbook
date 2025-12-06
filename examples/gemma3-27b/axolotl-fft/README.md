## Run instructions

### Update config.yaml 
- If pushing your checkpoints to Huggingface, make sure to update the `hub_model_id` in `config.yaml`, else comment out the lines with `hub_model_id` and `hub_strategy`. 
- If using Weights&Biases to track your run, update `use_wandb`, `wandb_project` and `wandb_entity` accordingly. 
- Hyper parameters, dataset processing and other important fields are defined in this yaml. Please go over all fields to make sure they work for your training use-case. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.