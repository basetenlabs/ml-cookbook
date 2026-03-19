#!/bin/bash
set -euo pipefail

# =============================================================================
# DeepSeek V3.1 Model Loader
# =============================================================================
# Downloads DeepSeek-V3.1-Base model to BT_TEAM_CACHE_DIR
# =============================================================================

# Install dependencies
pip install huggingface_hub hf_transfer

# Clone torchtitan for the download script
git clone https://github.com/aghilann/torchtitan
cd torchtitan

# Download the model assets
python scripts/download_hf_assets.py \
    --repo_id deepseek-ai/DeepSeek-V3.1-Base \
    --assets safetensors config tokenizer \
    --local_dir $BT_TEAM_CACHE_DIR

echo ""
echo "=============================================="
echo "Model downloaded to: $BT_TEAM_CACHE_DIR"
echo "=============================================="
echo ""
echo "Contents:"
ls -lah $BT_TEAM_CACHE_DIR
