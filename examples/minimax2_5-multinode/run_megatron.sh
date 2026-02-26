#!/usr/bin/env bash
set -euo pipefail

LOG_BASE_DIR="${BT_CHECKPOINT_DIR:-/mnt/ckpts}/debug_logs"
mkdir -p "${LOG_BASE_DIR}"
LOG_FILE="${LOG_BASE_DIR}/run-megatron-node-${BT_NODE_RANK:-0}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "==== run_megatron.sh start node=${BT_NODE_RANK:-0} $(date -Is) ===="
echo "log_file=${LOG_FILE}"

CACHE_ROOT="/tmp"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-cache"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch-extensions"
export NCCL_DEBUG="INFO"
export TORCH_DISTRIBUTED_DEBUG="OFF"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export NCCL_SOCKET_IFNAME="^docker0,lo"
export OMP_NUM_THREADS="4"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_COMPILE_DISABLE="1"
export TORCHINDUCTOR_COMPILE_THREADS="1"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

export MASTER_PORT="29500"

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
export CUDA_LAUNCH_BLOCKING=1


PY_BIN="$(command -v python3 || command -v python)"
if [[ -z "${PY_BIN}" ]]; then
  echo "No Python interpreter found in PATH." >&2
  exit 1
fi

SWIFT_VERSION="3.12.5"
"${PY_BIN}" -m pip install --upgrade pip
"${PY_BIN}" -m pip install "ms-swift[llm]==${SWIFT_VERSION}" datasets huggingface_hub
"${PY_BIN}" -m pip install "transformers==4.57.1" -U

# Training variables (edit these directly; all set to fixed defaults).
MODEL_ID="MiniMaxAI/MiniMax-M2.5"
DATASET_ID="winglian/pirate-ultrachat-10k"
DATASET_SPLIT="train"
CHECKPOINT_NAME="minimax-m2-5-megatron-lora"
RUN_NAME="minimax-m2-5-megatron-lora"
MODEL_ARG="${MODEL_ID}"

LORA_RANK=8
LORA_ALPHA=16
SPLIT_DATASET_RATIO=0.01
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
EXPERT_PARALLEL_SIZE=16
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2
MAX_EPOCHS=1
LR=2e-4
LR_DECAY_STYLE="constant"
LR_WARMUP_FRACTION=0.05
MIN_LR=1e-5
MAX_LENGTH=2048
NUM_WORKERS=8
DATASET_NUM_PROC=8
SAVE_FULL_MODEL="false"
SAVE_INTERVAL=5
LOG_INTERVAL=1
REPORT_TO="none"
MSSWIFT_COMPAT_MODE="true"


if [[ ! -d "${MODEL_ID}" ]]; then
  LOCAL_MODEL_DIR="${CACHE_ROOT}/model-snapshots/${MODEL_ID//\//__}"
  mkdir -p "${LOCAL_MODEL_DIR}"
  echo "Pre-downloading model snapshot to cache: ${LOCAL_MODEL_DIR}"
  MODEL_ID_ENV="${MODEL_ID}" LOCAL_MODEL_DIR_ENV="${LOCAL_MODEL_DIR}" "${PY_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID_ENV"],
    local_dir=os.environ["LOCAL_MODEL_DIR_ENV"],
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"snapshot_ready={os.environ['LOCAL_MODEL_DIR_ENV']}")
PY
  MODEL_ARG="${LOCAL_MODEL_DIR}"
fi

checkpoint_dir="${BT_CHECKPOINT_DIR:-/mnt/ckpts}/${CHECKPOINT_NAME}"
mkdir -p "${checkpoint_dir}"
printf '{}' > "${checkpoint_dir}/args.json"

# Megatron constraint:
# global_batch_size % (micro_batch_size * data_parallel_size) == 0
WORLD_SIZE=$(( ${BT_GROUP_SIZE:-1} * ${BT_NUM_GPUS:-1} ))
MODEL_PARALLEL_SIZE=$(( TENSOR_PARALLEL_SIZE * EXPERT_PARALLEL_SIZE ))
if (( MODEL_PARALLEL_SIZE <= 0 )); then
  MODEL_PARALLEL_SIZE=1
fi
DATA_PARALLEL_SIZE=$(( WORLD_SIZE / MODEL_PARALLEL_SIZE ))
if (( DATA_PARALLEL_SIZE <= 0 )); then
  DATA_PARALLEL_SIZE=1
fi
MIN_DIVISOR=$(( MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE ))
if (( MIN_DIVISOR <= 0 )); then
  MIN_DIVISOR=1
fi
if (( GLOBAL_BATCH_SIZE % MIN_DIVISOR != 0 )); then
  echo "Adjusting GLOBAL_BATCH_SIZE from ${GLOBAL_BATCH_SIZE} to ${MIN_DIVISOR} to satisfy Megatron divisibility."
  GLOBAL_BATCH_SIZE="${MIN_DIVISOR}"
fi

echo "Launching ms-swift Megatron SFT for ${MODEL_ID}"
echo "model_path=${MODEL_ARG}"
echo "checkpoint_dir=${checkpoint_dir}"
echo "BT_GROUP_SIZE=${BT_GROUP_SIZE} BT_NUM_GPUS=${BT_NUM_GPUS} BT_NODE_RANK=${BT_NODE_RANK}"
echo "MASTER_ADDR=${BT_LEADER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "ms_swift_version=${SWIFT_VERSION}"
echo "profile=working_config TP=${TENSOR_PARALLEL_SIZE} EP=${EXPERT_PARALLEL_SIZE} PP=${PIPELINE_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} LORA_RANK=${LORA_RANK} MAX_LENGTH=${MAX_LENGTH} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "MSSWIFT_COMPAT_MODE=${MSSWIFT_COMPAT_MODE}"

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
SPLIT_ARGS=()
if [[ "${DATASET_SPLIT}" != "train" ]]; then
  SPLIT_JSON="[{\"hf_dataset_id\": \"${DATASET_ID}\", \"split\": [\"${DATASET_SPLIT}\"]}]"
  SPLIT_ARGS=(--custom_dataset_info "${SPLIT_JSON}")
fi

COMPAT_ARGS=()
if [[ "${MSSWIFT_COMPAT_MODE}" == "true" ]]; then
  COMPAT_ARGS=(
    --overlap_grad_reduce false
    --overlap_param_gather false
    --use_distributed_optimizer false
  )
fi

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
NPROC_PER_NODE="${BT_NUM_GPUS}" \
NNODES="${BT_GROUP_SIZE}" \
NODE_RANK="${BT_NODE_RANK}" \
MASTER_ADDR="${BT_LEADER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
megatron sft \
  --model "${MODEL_ARG}" \
  --save "${checkpoint_dir}" \
  --dataset "${DATASET_ID}" \
  --template minimax_m2 \
  --check_model false \
  --load_safetensors true \
  --train_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --merge_lora "${SAVE_FULL_MODEL}" \
  --target_modules all-linear \
  --max_epochs "${MAX_EPOCHS}" \
  --lr_decay_style "${LR_DECAY_STYLE}" \
  --clip_grad 1.0 \
  --split_dataset_ratio "${SPLIT_DATASET_RATIO}" \
  --tensor_model_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline_model_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE}" \
  --expert_model_parallel_size "${EXPERT_PARALLEL_SIZE}" \
  --bf16 true \
  --loss_scale default \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --packing false \
  --cross_entropy_loss_fusion true \
  --recompute_granularity selective \
  --recompute_modules core_attn moe \
  --lr "${LR}" \
  --lr_warmup_fraction "${LR_WARMUP_FRACTION}" \
  --min_lr "${MIN_LR}" \
  --max_length "${MAX_LENGTH}" \
  --save_interval "${SAVE_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --num_workers "${NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --lazy_tokenize true \
  --load_from_cache_file true \
  --no_save_optim true \
  --no_save_rng true \
  --sequence_parallel true \
  --attention_backend flash \
  "${COMPAT_ARGS[@]}" \
  "${SPLIT_ARGS[@]}" \
  --use_hf 1
TRAIN_EXIT_CODE=$?
set -e



export CHECKPOINT_DIR="${checkpoint_dir}"
export HUB_MODEL_ID="baseten-admin/minimax-m2-5-megatron-lora"
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
