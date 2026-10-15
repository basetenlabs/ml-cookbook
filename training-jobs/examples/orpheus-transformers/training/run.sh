

set -e

accelerate launch --num_processes $BT_NUM_GPUS train.py