"""
Training Script for Programmatic Training API

This script reads configuration from environment variables set by config.py,
which in turn reads from runtime_config.json written by the training handler.
"""

import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer

# Read basic configuration from environment
MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
DATASET_ID = os.environ.get("DATASET_ID", "HuggingFaceH4/Multilingual-Thinking")

# Read advanced configuration from RUNTIME_CONFIG if available
runtime_config_str = os.environ.get("RUNTIME_CONFIG", "{}")
runtime_config = json.loads(runtime_config_str)

# Extract training parameters with defaults
training_params = runtime_config.get("training_params", {})
LEARNING_RATE = training_params.get("learning_rate", 2e-4)
NUM_EPOCHS = training_params.get("num_epochs", 1)
BATCH_SIZE = training_params.get("batch_size", 4)
MAX_SEQ_LENGTH = training_params.get("max_seq_length", 2048)

# Extract LoRA parameters with defaults
lora_params = runtime_config.get("lora_params", {})
LORA_R = lora_params.get("r", 8)
LORA_ALPHA = lora_params.get("alpha", 16)

print(f"Training Configuration:")
print(f"  Model: {MODEL_ID}")
print(f"  Dataset: {DATASET_ID}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"  LoRA Rank: {LORA_R}")
print(f"  LoRA Alpha: {LORA_ALPHA}")

# Load dataset and tokenizer
dataset = load_dataset(DATASET_ID, split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure model quantization
quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

# Configure LoRA
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",
    # Customize target_parameters based on your model architecture
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

# Configure training
output_dir = os.getenv("BT_CHECKPOINT_DIR", "./checkpoints")
training_args = SFTConfig(
    learning_rate=LEARNING_RATE,
    gradient_checkpointing=True,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    max_length=MAX_SEQ_LENGTH,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir=output_dir,
    report_to="none",
    push_to_hub=False,
)

# Train
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()

# Save model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")

