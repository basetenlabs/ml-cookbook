import torch
from nemo import lightning as nl
import nemo_run as run
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import pytorch_lightning as pl
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from nemo.utils.import_utils import safe_import
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

_, HAVE_TE = safe_import("transformer_engine")


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None, # autotokenizer?
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask
        # self.create_attention_mask = create_attention_mask or not HAVE_TE

        if tokenizer is None:
            from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
            self.tokenizer = get_nmt_tokenizer(
                "megatron", "GPT2BPETokenizer", vocab_file=vocab_file, merges_file=merges_file
            )
        else:
            self.tokenizer = tokenizer

        # self.data_sampler = MegatronDataSampler(
        #     seq_len=self.seq_length,
        #     micro_batch_size=self.micro_batch_size,
        #     global_batch_size=self.global_batch_size,
        #     rampup_batch_size=rampup_batch_size,
        # )

    def setup(self, stage: str = "") -> None:
        """
        Setup the data module.
        """
        self._train_ds = load_dataset(self.dataset_name, split="train")
        self._validation_ds = load_dataset(self.dataset_name, split="validation")
        self._test_ds = load_dataset(self.dataset_name, split="test")

    def train_dataloader(self):
        """
        Get the train dataloader.
        """
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self):
        """
        Get the validation dataloader.
        """
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self):
        """
        Get the test dataloader.
        """
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )

    def reconfigure_limit_batches(self):
        """
        Reconfigure trainer.limit_train_batches and trainer.limit_val_batches in terms of num of microbatches.
        """
        from nemo.collections.llm.gpt.data.utils import _reconfigure_limit_batches

        # Override limit_train_batches in terms of num of microbatches
        self.trainer.limit_train_batches = _reconfigure_limit_batches(self.trainer.limit_train_batches, self._train_ds)
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting
        #   in between a step
        self.trainer.limit_val_batches = _reconfigure_limit_batches(
            self.trainer.limit_val_batches, self._validation_ds
        )

        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

        except (ImportError, ModuleNotFoundError):
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        # Override num sanity steps to be a multiple of num of microbatches
        self.trainer.num_sanity_val_steps *= get_num_microbatches()

if __name__ == "__main__":
    seq_length = 2048
    global_batch_size = 16

    ## setup the dummy dataset
    # data = llm.MockDataModule(seq_length=seq_length, global_batch_size=global_batch_size)
    tokenizer = get_nmt_tokenizer(
        "megatron", "GPT2BPETokenizer"
    )
    data = llm.HFDatasetDataModule(path_or_dataset="nvidia/OpenMathInstruct-1", tokenizer=tokenizer)
    # data = CustomDataModule(
    #     dataset_name="tatsu-lab/alpaca", 
    #     seq_length=seq_length, 
    #     global_batch_size=global_batch_size,
    #     # tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B"),
    #     )

    ## initialize a small GPT model
    gpt_config = llm.GPTConfig(
        num_layers=6,
        hidden_size=384,
        ffn_hidden_size=1536,
        num_attention_heads=6,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )

    
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)
    # model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=2, ## you can change the number of devices to suit your setup
        max_steps=200,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="test_logdir", ## logs and checkpoints will be written here
    )
    print("Done initializing logger ...")
    print("Beginning training...")

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )

    # recipe = run.Partial(
    #     llm.train,
    #     model=model,
    #     data=data,
    #     trainer=trainer,
    #     log=nemo_logger,
    #     tokenizer='data',
    #     optim=opt,
    # )

    # run.run(recipe, executor=run.LocalExecutor())

