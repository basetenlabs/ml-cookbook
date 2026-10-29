#!/bin/bash
set -euxo pipefail
export HF_HOME=$BT_PROJECT_CACHE_DIR/huggingface TORCHDYNAMO_DISABLE=1
HERE=$PWD
DATA=/tmp/data
STACK=/tmp
mkdir -p $DATA

# hub pins match the model repo's constraints.txt
pip install -q huggingface_hub==2.2.0 httpx2==2.13.1 brotli==1.2.0 datasets==5.1.0
MODEL=$(python -c "from huggingface_hub import snapshot_download as d; print(d('baselabs/joey-1', revision='3903da0d1252277671a0865daeec29c1965f1cb5'))")
bash $MODEL/setup.sh $STACK
# Joey-1's input embedding and LM head share one weight, which OneJev's FSDP sharding refuses without this patch
git -C $STACK/OneJev apply $HERE/tied_embeddings.patch
python $HERE/prepare_data.py $DATA

# OneJev has no 2B config; the 9B one differs only in its base model, which --model replaces
# --attention noncausal and --readout slot_lm match how Joey-1 was trained
cd $STACK/OneJev
torchrun --nproc-per-node=$BT_NUM_GPUS -m train.sft --config train/configs/onejev_9b_full.yaml \
  --model $MODEL --train $DATA/train.jsonl --media-root $DATA --limit ${ROWS:-2000} --lr 1e-6 \
  --attention noncausal --readout slot_lm --eval test=$DATA/test.jsonl --eval-limit 500 --eval-at-start \
  --save-every 0 --runs-dir $BT_CHECKPOINT_DIR --name joey-1-sft
