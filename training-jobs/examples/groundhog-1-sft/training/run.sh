#!/bin/bash
set -euxo pipefail
# the pytorch 2.11 image uses Debian's system Python, which refuses pip installs without this
export HF_HOME=$BT_PROJECT_CACHE_DIR/huggingface TORCHDYNAMO_DISABLE=1 PIP_BREAK_SYSTEM_PACKAGES=1
HERE=$PWD
DATA=/tmp/data
STACK=/tmp
mkdir -p $DATA

# hub pins match the model repo's constraints.txt
pip install -q huggingface_hub==1.33.0 brotli==1.2.0 datasets==5.1.0
MODEL=$(python -c "from huggingface_hub import snapshot_download as d; print(d('baselabs/groundhog-1', revision='f4e4a46e0f7f14499dc1e5b080e3b50f2333ded8'))")
bash $MODEL/setup.sh $STACK
# OneJev holds a full fp32 copy of the model on every GPU while it shards, which overflows an 80 GB H100 without this patch
git -C $STACK/OneJev apply $HERE/fsdp_setup_memory.patch
python $HERE/prepare_data.py $DATA

# OneJev has no Gemma config; the 27B one differs only in its base model, which --model replaces
# --attention noncausal and --readout slot_lm match how Groundhog-1 was trained, and it kept its routers, norms and
# embeddings frozen
cd $STACK/OneJev
torchrun --nproc-per-node=$BT_NUM_GPUS -m train.sft --config train/configs/onejev_27b_full.yaml \
  --model $MODEL --train $DATA/train.jsonl --media-root $DATA --limit ${ROWS:-2000} --lr 1e-6 \
  --attention noncausal --readout slot_lm --freeze-router --freeze-norms-embeds \
  --eval test=$DATA/test.jsonl --eval-limit 500 --eval-at-start \
  --save-every 0 --runs-dir $BT_CHECKPOINT_DIR --name groundhog-1-sft
