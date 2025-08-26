export CHECKPOINT_HF=baseten-admin/whisper-larger-v3-turbo-minirun3

# vllm serve $CHECKPOINT_HF --served-model-name gemma3 --max-model-len 32000 --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --trust-remote-code --dtype bfloat16
vllm serve $CHECKPOINT_HF --dtype auto