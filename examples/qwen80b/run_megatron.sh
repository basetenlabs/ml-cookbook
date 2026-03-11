#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HF token is not set. An HF token is required to upload final checkpoints to Hugging Face. Configure Baseten secret 'hf_access_token' and map it to HF_TOKEN in config.py."
  exit 1
fi

CACHE_ROOT="/tmp"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-cache"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch-extensions"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export NCCL_SOCKET_IFNAME="^docker0,lo"
export OMP_NUM_THREADS="4"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_COMPILE_THREADS="1"
export TORCH_DISABLE_ADDR2LINE=1
export MASTER_PORT="29500"

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

SWIFT_VERSION="3.12.5"
python -m pip install --upgrade pip
python -m pip install "ms-swift[llm]==${SWIFT_VERSION}" datasets huggingface_hub
python -m pip install "transformers==4.57.1" -U

MODEL_ID="Qwen/Qwen3-Next-80B-A3B-Instruct"
CHECKPOINT_NAME="qwen80b-instruct-megatron-lora"

checkpoint_dir="${BT_CHECKPOINT_DIR:-/mnt/ckpts}/${CHECKPOINT_NAME}"
mkdir -p "${checkpoint_dir}"
printf '{}' > "${checkpoint_dir}/args.json"

echo "Launching ms-swift Megatron SFT for ${MODEL_ID}"
echo "checkpoint_dir=${checkpoint_dir}"
echo "BT_GROUP_SIZE=${BT_GROUP_SIZE} BT_NUM_GPUS=${BT_NUM_GPUS} BT_NODE_RANK=${BT_NODE_RANK}"
echo "MASTER_ADDR=${BT_LEADER_ADDR} MASTER_PORT=${MASTER_PORT}"

# ms-swift may write checkpoints into timestamped subdirs (e.g., v0-YYYYMMDD-HHMMSS).
# Keep args.json mirrored there to avoid checkpoint save failures.
ARGS_SYNC_PID=""
sync_args_json_loop() {
  while true; do
    for run_dir in "${checkpoint_dir}"/v*-*; do
      if [[ -d "${run_dir}" ]] && [[ ! -f "${run_dir}/args.json" ]]; then
        cp "${checkpoint_dir}/args.json" "${run_dir}/args.json" || true
      fi
    done
    sleep 1
  done
}

sync_args_json_loop &
ARGS_SYNC_PID=$!
cleanup_args_sync() {
  if [[ -n "${ARGS_SYNC_PID}" ]]; then
    kill "${ARGS_SYNC_PID}" 2>/dev/null || true
  fi
}
trap cleanup_args_sync EXIT

set +e
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
NPROC_PER_NODE="${BT_NUM_GPUS}" \
NNODES="${BT_GROUP_SIZE}" \
NODE_RANK="${BT_NODE_RANK}" \
MASTER_ADDR="${BT_LEADER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
megatron sft \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --model_type qwen3_next \
  --save "${checkpoint_dir}" \
  --dataset winglian/pirate-ultrachat-10k \
  --template minimax_m2 \
  --check_model false \
  --load_safetensors true \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --merge_lora false \
  --target_modules all-linear \
  --max_epochs 1 \
  --lr_decay_style constant \
  --clip_grad 1.0 \
  --split_dataset_ratio 0.01 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --context_parallel_size 1 \
  --expert_model_parallel_size 8 \
  --bf16 true \
  --loss_scale default \
  --micro_batch_size 1 \
  --global_batch_size 8 \
  --packing false \
  --cross_entropy_loss_fusion true \
  --recompute_granularity selective \
  --recompute_modules core_attn moe \
  --lr 2e-4 \
  --lr_warmup_fraction 0.05 \
  --min_lr 1e-5 \
  --max_length 2048 \
  --save_interval 5 \
  --log_interval 1 \
  --num_workers 8 \
  --dataset_num_proc 8 \
  --lazy_tokenize true \
  --load_from_cache_file true \
  --no_save_optim true \
  --no_save_rng true \
  --sequence_parallel true \
  --attention_backend flash \
  --overlap_grad_reduce false \
  --overlap_param_gather false \
  --use_distributed_optimizer false \
  --use_hf 1
TRAIN_EXIT_CODE=$?
set -e

export CHECKPOINT_DIR="${checkpoint_dir}"
export HUB_MODEL_ID="baseten-admin/qwen80b-instruct-megatron-lora"
NODE1_UPLOAD_DONE_MARKER="${CHECKPOINT_DIR}/.node1_upload_done"
# Final checkpoint upload to Hugging Face Hub from rank 0.
if [[ "${BT_NODE_RANK}" == "1" ]]; then
  echo "Starting final HF Hub upload from ${CHECKPOINT_DIR} to ${HUB_MODEL_ID}..."
  "${PY_BIN}" - <<'PY'
import datetime
import os
from huggingface_hub import HfApi

checkpoint_dir = os.environ["CHECKPOINT_DIR"]
repo_id = os.environ["HUB_MODEL_ID"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=checkpoint_dir,
    commit_message=f"Final checkpoint upload {datetime.datetime.utcnow().isoformat()}Z",
)
print(f"hf_upload_complete repo={repo_id} folder={checkpoint_dir}")
PY
  touch "${NODE1_UPLOAD_DONE_MARKER}"
fi

if [[ "${BT_NODE_RANK}" == "0" ]]; then
  NODE0_WAIT_TIMEOUT_SECONDS="${NODE0_WAIT_TIMEOUT_SECONDS:-3600}"
  NODE0_WAIT_POLL_SECONDS="${NODE0_WAIT_POLL_SECONDS:-5}"
  echo "Node 0 waiting for node 1 upload marker: ${NODE1_UPLOAD_DONE_MARKER} (timeout ${NODE0_WAIT_TIMEOUT_SECONDS}s)"
  waited=0
  while [[ ! -f "${NODE1_UPLOAD_DONE_MARKER}" ]]; do
    sleep "${NODE0_WAIT_POLL_SECONDS}"
    waited=$(( waited + NODE0_WAIT_POLL_SECONDS ))
    if (( waited >= NODE0_WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for node 1 upload completion marker."
      exit 1
    fi
  done
  echo "Node 1 upload marker found."
fi

echo "[rank ${BT_NODE_RANK}] megatron exit code=${TRAIN_EXIT_CODE}"
if [[ "${TRAIN_EXIT_CODE}" -ne 0 ]]; then
  if [[ -d "${checkpoint_dir}" ]] && find "${checkpoint_dir}" -name "*.safetensors" -type f | grep -q .; then
    echo "Training exited ${TRAIN_EXIT_CODE}, but safetensors exist in ${checkpoint_dir}; treating as success."
    exit 0
  fi
  exit "${TRAIN_EXIT_CODE}"
fi
