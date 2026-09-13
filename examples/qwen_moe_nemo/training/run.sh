#!/usr/bin/env bash
set -euo pipefail

export NEMO_CACHE_DIR=$BT_RW_CACHE_DIR
export NEMO_MODELS_CACHE=$NEMO_CACHE_DIR/nemo_models


nemo llm import model=qwen3_30b_a3b source="hf://Qwen/Qwen3-30B-A3B" -y
nemo llm finetune --factory "qwen3_30b_a3b(peft_scheme=none)" -y # uses dummy dataset