## Run instructions

### Update config.py 
- Make sure the environment variables in `config.py` is updated with the same name of secrets as saved in Baseten secrets.
- `train.py` has the data processing calls, training code and hyper parameters defined. Update datasets, flags etc here or pass them as arguments through `run.sh` to change them. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.