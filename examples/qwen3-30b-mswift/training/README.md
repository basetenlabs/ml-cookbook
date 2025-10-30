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

### Accessing checkpoints
To get the location of your checkpoints, run `truss train get_checkpoint_urls --job-id your-job-id`, this downloads a file containing information about your checkpoint storage locations.

### Troubleshooting CUDA OOM Errors
CUDA out-of-memory (OOM) errors are often buried by Megatron and may manifest as NCCL errors instead. If you encounter errors like `RuntimeError: NCCL Error 1: unhandled cuda error … group.reduce_scatter_tensor_coalesced(outputs, inputs, reduce_opts)`, this typically indicates that your GPU memory usage is too high. In such cases, you should tweak some of the launch commands in `run_1node.sh` to reduce memory consumption (e.g., adjust batch size, sequence length, or model parallelism settings).