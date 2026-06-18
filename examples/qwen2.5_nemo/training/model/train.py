import argparse
import os
import nemo_run as run
import lightning.pytorch as pl
from nemo.collections import llm

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

### Dataset
from data import BespokeDataModule

def bespoke(seq_length: int = 8000, micro_batch_size: int = 1, global_batch_size: int = 4, num_workers: int = 1) -> run.Config[pl.LightningDataModule]:
    return run.Config(BespokeDataModule, seq_length=seq_length, micro_batch_size=micro_batch_size, global_batch_size=global_batch_size, num_workers=num_workers)

import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import torch
import lightning.pytorch as pl
from pathlib import Path
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed

# Configure the trainer
def trainer(devices: int = 4, max_steps: int = 300, tensor_model_parallel_size: int = 4, log_every_n_steps: int = 1) -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )
    trainer = run.Config(
        nl.Trainer,
        devices=devices,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=bf16_mixed(),
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=0,
        val_check_interval=0,
        num_sanity_val_steps=0,
    )
    return trainer

# Configure the logger
def logger(name: str = "qwen_sft", log_dir: str = "./results", checkpoint_every_n_steps: int = 100, save_top_k: int = 1) -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        every_n_train_steps=checkpoint_every_n_steps,
        monitor="reduced_train_loss",
        save_top_k=save_top_k,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return run.Config(
        nl.NeMoLogger,
        name=name,
        log_dir=log_dir,
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=None
    )

# Configure the optimizer
def adam_with_cosine_annealing(lr: float = 2e-5, adam_beta2: float = 0.98, clip_grad: float = 1.0) -> run.Config[nl.OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=lr,
        adam_beta2=adam_beta2,
        use_distributed_optimizer=True,
        clip_grad=clip_grad,
        bf16=True,
    )
    return run.Config(
        nl.MegatronOptimizerModule,
        config=opt_cfg
    )

# Configure the model
def qwen() -> run.Config[pl.LightningModule]:
    return run.Config(llm.Qwen2Model, config=run.Config(llm.Qwen2Config7B))

# Configure the resume
def resume(model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig,
            path=f"nemo://{model_id}"
        ),
        resume_if_exists=True,
    )

def configure_finetuning_recipe(args):
    return run.Partial(
        llm.finetune,
        model=qwen(),
        trainer=trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            log_every_n_steps=args.log_every_n_steps
        ),
        data=bespoke(
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            num_workers=args.num_workers
        ),
        log=logger(
            name=args.experiment_name,
            log_dir=args.log_dir,
            checkpoint_every_n_steps=args.checkpoint_every_n_steps,
            save_top_k=args.save_top_k
        ),
        optim=adam_with_cosine_annealing(
            lr=args.learning_rate,
            adam_beta2=args.adam_beta2,
            clip_grad=args.clip_grad
        ),
        resume=resume(args.model_id),
    )

def local_executor_torchrun(nodes: int = 1, devices: int = 4) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Qwen2.5 Fine-tuning Script")
    output_dir = os.getenv("BT_CHECKPOINT_DIR", "./results")
    
    # Training parameters
    parser.add_argument("--devices", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum training steps")
    parser.add_argument("--tensor_model_parallel_size", type=int, default=4, help="Tensor model parallel size")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Log every N steps")
    
    # Data parameters
    parser.add_argument("--seq_length", type=int, default=8192, help="Sequence length")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--global_batch_size", type=int, default=8, help="Global batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")
    
    # Optimizer parameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="Adam beta2 parameter")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    
    # Logging and checkpointing
    parser.add_argument("--experiment_name", type=str, default="qwen_sft", help="Experiment name")
    parser.add_argument("--log_dir", type=str, default=output_dir, help="Logging directory")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--save_top_k", type=int, default=1, help="Save top K checkpoints")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Initial checkpoint path")
    
    # Executor parameters
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print(f"Starting training with configuration:")
    print(f"  Devices: {args.devices}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Global batch size: {args.global_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Experiment name: {args.experiment_name}")
    
    run.run(
        configure_finetuning_recipe(args), 
        executor=local_executor_torchrun(nodes=args.nodes, devices=args.devices)
    )