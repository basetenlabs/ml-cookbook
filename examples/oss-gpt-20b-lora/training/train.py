import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
import torch

from trl import SFTConfig, SFTTrainer

MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")  # The model to fine-tune
DATASET_ID = os.environ.get("DATASET_ID", "HuggingFaceH4/Multilingual-Thinking")  # The dataset to use for fine-tuning

dataset = load_dataset(DATASET_ID, split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="gpt-oss-20b-multilingual-reasoner",
    report_to="trackio",
    push_to_hub=False,
)


trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()

trainer.save_model(training_args.output_dir)
# Push the trained model in output_dir to a Hugging Face model repo
hf_write_loc = os.environ.get("HF_WRITE_LOC", "baseten-admin/gpt-oss-20b-multilingual-reasoner")
trainer.push_to_hub(hf_write_loc)