import nemo_run as run
import lightning.pytorch as pl
from nemo.collections import llm

# llm.import_ckpt is the nemo2 API for converting Hugging Face checkpoint to NeMo format
# example python usage:
# llm.import_ckpt(model=llm.llama3_8b.model(), source="hf://meta-llama/Meta-Llama-3-8B")
#
# We use run.Partial to configure this function
def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=llm.qwen2_7b.model(),
        source="hf://Qwen/Qwen2.5-7B-Instruct",
        overwrite=True,
    )

# configure your function
import_ckpt = configure_checkpoint_conversion()
# define your executor
local_executor = run.LocalExecutor()

# run your experiment
run.run(import_ckpt, executor=local_executor)

### Dataset
from bespoke import BespokeDataModule

def bespoke() -> run.Config[pl.LightningDataModule]:
    return run.Config(BespokeDataModule, seq_length=16384, micro_batch_size=1, global_batch_size=32, num_workers=1)

import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import torch
import lightning.pytorch as pl
from pathlib import Path
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed


# Configure the trainer
# we use 4 GPUs for training and set the max_steps to 300.
def trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=4,
    )
    trainer = run.Config(
        nl.Trainer,
        devices=4,
        max_steps=300,
        accelerator="gpu",
        strategy=strategy,
        plugins=bf16_mixed(),
        log_every_n_steps=1,
        limit_val_batches=0,
        val_check_interval=0,
        num_sanity_val_steps=0,
    )
    return trainer

# Configure the logger
# Here, we configure the log interval to 100 steps and save the model every 100 steps. you can change these parameters as needed.
def logger() -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        every_n_train_steps=100,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return run.Config(
        nl.NeMoLogger,
        name="qwen_sft",
        log_dir="./results",
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=None
    )


# Configure the optimizer
# We use the distributed Adam optimizer and pass in the OptimizerConfig.
def adam_with_cosine_annealing() -> run.Config[nl.OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=2e-5,
        adam_beta2=0.98,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
    )
    return run.Config(
        nl.MegatronOptimizerModule,
        config=opt_cfg
    )

# Configure the model
# We use Qwen2Config7B to configure the model.
def qwen() -> run.Config[pl.LightningModule]:
    return run.Config(llm.Qwen2Model, config=run.Config(llm.Qwen2Config7B))

# Configure the resume
def resume() -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig,
            path="nemo://Qwen/Qwen2.5-7B-Instruct"
        ),
        resume_if_exists=True,
    )

def configure_finetuning_recipe():
    return run.Partial(
        llm.finetune,
        model=qwen(),
        trainer=trainer(),
        data=bespoke(),
        log=logger(),
        optim=adam_with_cosine_annealing(),
        resume=resume(),
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

if __name__ == '__main__':
    run.run(configure_finetuning_recipe(), executor=local_executor_torchrun())
