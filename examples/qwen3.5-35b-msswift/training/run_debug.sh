#!/bin/bash
set -eux

# ---------------------------------------------------------------------------
# Debug mode: install/verify deps, then sleep forever so we can SSH in and
# run megatron sft commands by hand. Same dep install logic as run.sh, just
# without the training launch at the end.
# ---------------------------------------------------------------------------

PKG_DIR=$BT_PROJECT_CACHE_DIR/qwen3_5_packages
export PYTHONPATH=$PKG_DIR:${PYTHONPATH:-}

SWIFT_OK=$(python -c "import swift; print(swift.__version__)" 2>/dev/null || echo "0.0.0")
TF_OK=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "0.0.0")
HUB_OK=$(python -c "from huggingface_hub import is_offline_mode; print('ok')" 2>/dev/null || echo "no")
BRIDGE_OK=$(python -c "import importlib.metadata; print(importlib.metadata.version('mcore-bridge'))" 2>/dev/null || echo "no")
if [ "$(printf '%s\n4.1.0' "$SWIFT_OK" | sort -V | head -1)" != "4.1.0" ] \
   || ! python -c "import transformers; assert transformers.__version__.startswith('5.2.')" 2>/dev/null \
   || [ "$HUB_OK" != "ok" ] \
   || [ "$BRIDGE_OK" = "no" ]; then
    echo "Upgrading deps (swift=$SWIFT_OK tf=$TF_OK hub_ok=$HUB_OK bridge=$BRIDGE_OK)"
    pip install --target=$PKG_DIR --no-deps \
        "ms-swift>=4.1.0" \
        "transformers==5.2.*" \
        "huggingface_hub>=1.3.0,<2.0" \
        "tokenizers>=0.22.0,<=0.23.0" \
        "safetensors>=0.4.3" \
        "accelerate>=1.1.0" \
        "peft>=0.13" \
        "liger-kernel" \
        "qwen_vl_utils>=0.0.14" \
        "mcore-bridge>=1.0.2" \
        "torchao>=0.16" \
        "tilelang" \
        "megatron-core>=0.16"
fi
if ! python -c "from fla.ops.utils import *; import causal_conv1d" 2>/dev/null; then
    pip install --target=$PKG_DIR --no-deps \
        "git+https://github.com/fla-org/flash-linear-attention.git"
    pip install --target=$PKG_DIR --no-deps --no-build-isolation \
        "git+https://github.com/Dao-AILab/causal-conv1d"
fi

export HF_HOME=$BT_PROJECT_CACHE_DIR/huggingface
mkdir -p $HF_HOME/hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B')
"

# Drop a helper file in $HOME so anyone SSHing in can `source` it.
cat > $HOME/qwen35_env.sh <<EOF
export PYTHONPATH=$PKG_DIR:\${PYTHONPATH:-}
export HF_HOME=$HF_HOME
export USE_MCORE_GDN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=\${BT_NUM_GPUS:-8}
export NNODES=\${BT_GROUP_SIZE:-1}
export NODE_RANK=\${BT_NODE_RANK:-0}
export MASTER_ADDR=\${BT_LEADER_ADDR:-127.0.0.1}
echo "Qwen3.5 debug env loaded. Try: megatron sft --help"
EOF
chmod +x $HOME/qwen35_env.sh

echo "===================================================================="
echo "Debug pod ready. SSH in, then run:  source ~/qwen35_env.sh"
echo "===================================================================="

# Sleep forever so the pod stays up for SSH.
sleep infinity
