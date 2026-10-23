#!/bin/bash
set -euxo pipefail
export HF_HOME=$BT_PROJECT_CACHE_DIR/huggingface TORCHDYNAMO_DISABLE=1
HERE=$PWD
DATA=/tmp/data
STACK=/tmp
mkdir -p $DATA

# hub pins match the model repo's constraints.txt
pip install -q huggingface_hub==2.2.0 httpx2==2.13.1 brotli==1.2.0 datasets==5.1.0
MODEL=$(python -c "from huggingface_hub import snapshot_download as d; print(d('baselabs/elephant-1'))")
bash $MODEL/setup.sh $STACK
python $HERE/prepare_data.py $DATA

# --attention noncausal and --readout slot_lm match how Elephant-1 was trained
cd $STACK/OneJev
torchrun --nproc-per-node=$BT_NUM_GPUS -m train.sft --config train/configs/onejev_9b_full.yaml \
  --model $MODEL --train $DATA/train.jsonl --media-root $DATA --limit ${ROWS:-2000} --lr 1e-6 \
  --attention noncausal --readout slot_lm --eval test=$DATA/test.jsonl --eval-limit 500 --eval-at-start \
  --save-every 0 --runs-dir $BT_CHECKPOINT_DIR --name elephant-1-sft
