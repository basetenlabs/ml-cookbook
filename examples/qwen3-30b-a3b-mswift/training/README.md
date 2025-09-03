## Run instructions

### Update config.py
Update `config.py` with environment variables to match your Baseten secrets. 

### Update run_1node.sh 
- If using Weights&Biases to track your run, update `wandb_project` and `wandb_exp_name` accordingly, or comment out these lines.
- Hyper parameters, dataset processing and other important fields are defined in this script. Please go over all fields to make sure they work for your training use-case. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.