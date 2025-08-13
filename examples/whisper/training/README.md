## Run instructions

### Update config.py 
- Make sure the environment variables in `config.py` is updated with the same name of secrets as saved in Baseten secrets.
- In `train.py`, where all the data processing and training code is present, make sure the data processing function `load_and_process_common_accent_dataset` or `load_and_process_dataset` processes your dataset as desired. This file has 2 examples - `load_and_process_common_accent_dataset` or `load_and_process_dataset`
    - This is also where hyper parameters for training are defined. Refer to `run.sh` to change them. 

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.