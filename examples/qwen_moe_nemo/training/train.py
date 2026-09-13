from nemo.collections import llm
import nemo_run as run
from nemo.collections.llm.gpt.data.hf_dataset import HFDatasetDataModule

# import lightning.pytorch as pl # data module has to be pl.LightningDataModule

# """
# export NEMO_CACHE_DIR=$BT_RW_CACHE_DIR
# export NEMO_MODELS_CACHE=$NEMO_CACHE_DIR/nemo_models

# Examples:
# >>> from nemo.collections import llm
# >>> from nemo import lightning as nl
# >>> model = llm.MistralModel()
# >>> data = llm.SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)
# >>> precision = nl.MegatronMixedPrecision(precision="bf16-mixed")
# >>> trainer = nl.Trainer(strategy=nl.MegatronStrategy(tensor_model_parallel_size=2), plugins=precision)
# >>> llm.finetune(model, data, trainer, peft=llm.peft.LoRA()])
# """

model = llm.Qwen3Model(llm.Qwen3Config30B_A3B())

llm.import_ckpt(model=model, source="hf://Qwen/Qwen3-30B-A3B")



# recipe = run.Partial(
#         llm.finetune,
#         model=model,
#         trainer=default_finetune_trainer(
#             num_nodes=num_nodes,
#             num_gpus_per_node=num_gpus_per_node,
#         ),
#         data=HFDatasetDataModule(path="tatsu-lab/alpaca", global_batch_size=16, micro_batch_size=2),
#         log=default_finetune_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
#         optim=distributed_fused_adam_with_cosine_annealing(max_lr=1e-4, min_lr=0, warmup_steps=50, adam_beta2=0.98),
#         resume=nemo_resume(resume_path),
#         tokenizer=tokenizer,
# )



# return recipe

recipe = llm.qwen3_30b_a3b.finetune_recipe(
    name="qwen3_30b_a3b_ft_nolora", 
    dir="checkpoints", 
    num_nodes=1, 
    num_gpus_per_node=8, 
    # peft_scheme='lora', 
    packed_sequence=False
    )

# recipe.data = HFDatasetDataModule(path_or_dataset="tatsu-lab/alpaca", global_batch_size=16, micro_batch_size=2)
# recipe.data = HFDatasetDataModule(path_or_dataset="tatsu-lab/alpaca")

recipe.data = llm.HFDatasetDataModule(path_or_dataset="nvidia/OpenMathInstruct-1")


run.run(recipe, executor=run.LocalExecutor())

