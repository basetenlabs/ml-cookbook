# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -eux

python3 prepare_dataset.py --local_dir /workspace/data/ocaml/

HF_HOME=$BT_RW_CACHE_DIR/huggingface
huggingface-cli download Qwen/Qwen3-8B

python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=reward_function.py \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/data/ocaml/train.parquet \
    data.val_files=/workspace/data/ocaml/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='OCaml Specialist' \
    trainer.experiment_name=$BT_TRAINING_JOB_NAME \
    trainer.n_gpus_per_node=$BT_NUM_GPUS \
    trainer.nnodes=$BT_GROUP_SIZE \
    trainer.default_local_dir=$BT_CHECKPOINT_DIR \
    trainer.save_freq=32 \
    trainer.test_freq=8 \
    trainer.total_epochs=1 $@ \
    algorithm.rollout_is_threshold=2.0 \
    algorithm.rollout_is=true \
    algorithm.rollout_is_level=token \
    algorithm.rollout_is_mode=truncate \

for checkpoint_dir in $BT_CHECKPOINT_DIR/global_step_*/; do
    if [ -d "$checkpoint_dir/actor" ]; then
        echo "Merging actor model from $(basename $checkpoint_dir)..."
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$checkpoint_dir/actor" \
            --target_dir "$checkpoint_dir/actor_hf"
    else
        echo "No actor directory found in $(basename $checkpoint_dir), skipping..."
    fi
done