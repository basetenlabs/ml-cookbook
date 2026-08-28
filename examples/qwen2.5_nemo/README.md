## Run instructions

### Update run.sh 
- Any changes to hyperparams used during training can be passed via flags to `model/train.py`. More details about flags are in `train.py`.
- All data processing in this example is in `model/data.py`. Repace this, or add data modules that reflect your dataset. Also remember to update `train.py` to use the correct data module. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.