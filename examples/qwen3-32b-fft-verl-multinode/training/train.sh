# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -eux

python3 prepare_dataset.py --local_dir /workspace/data/ocaml/

HF_HOME=$BT_RW_CACHE_DIR/huggingface
huggingface-cli download Qwen/Qwen3-32B

temperature=1.0
clip_ratio_low=0.2
clip_ratio_high=0.28
use_dynamic_bsz=True

ckpt_dir=${SHARED_CHECKPOINT_DIR}/${BT_TRAINING_JOB_NAME}
echo "Checkpoint directory: $ckpt_dir"

echo "Starting training"
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=reward_function.py \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/data/ocaml/train.parquet \
    data.val_files=/workspace/data/ocaml/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
    trainer.default_local_dir=$ckpt_dir \
    trainer.save_freq=2 \
    trainer.test_freq=2 \
    trainer.total_epochs=10 $@ \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is=token 