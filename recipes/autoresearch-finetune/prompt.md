Optimize val_loss by editing training/run.sh and submitting experiments to Baseten.

To run an experiment: edit training/run.sh (change any megatron sft flags), then submit with `truss train push training/config.py --non-interactive`, then monitor with `truss train logs --job-id <id> --non-interactive --tail` and read val_loss from the output block at the end of the job.

Keep improvements (git commit), discard regressions (git reset). Don't modify training/config.py or the results parsing block at the end of run.sh.
Run up to 10 experiments. You may run up to 2 jobs in parallel.
