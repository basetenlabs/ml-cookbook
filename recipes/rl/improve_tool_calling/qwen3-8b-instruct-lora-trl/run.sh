set -eux

pip install -U bitsandbytes trl transformers datasets peft accelerate

python train.py