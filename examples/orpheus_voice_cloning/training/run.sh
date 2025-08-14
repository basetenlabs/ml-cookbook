

set -e

wandb login $WANDB_API_KEY

accelerate launch --num_processes $BT_NUM_GPUS train.py