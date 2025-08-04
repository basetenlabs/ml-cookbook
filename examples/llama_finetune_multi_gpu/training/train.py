# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Define model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Fix tokenizer padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading dataset...")
dataset = load_dataset("nvidia/OpenMathInstruct-1")

def preprocess_function(examples):
    """
    Fixed preprocessing function that handles batched data properly
    """
    conversations = []
    
    # Handle both single examples and batches
    if isinstance(examples['question'], str):  # Single example
        questions = [examples['question']]
        solutions = [examples['generated_solution']]
    else: # Batch of examples
        questions = examples['question']
        solutions = examples['generated_solution']
    
    for question, solution in zip(questions, solutions):
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution}
        ]
        
        # Apply chat template and tokenize
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False  # We want the full conversation including response
        )
        conversations.append(formatted_text)
    
    # Tokenize all conversations
    tokenized = tokenizer(
        conversations,
        truncation=True,
        padding=False,  # We'll pad in data collator
        max_length=4096,  # Adjust based on your needs
        return_tensors=None  # Return lists, not tensors
    )
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Preprocessing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=dataset["train"].column_names,  # Remove original columns
    desc="Tokenizing dataset"
)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# Define training arguments - optimized for full fine-tuning
training_args = TrainingArguments(
    output_dir="./llama-math-finetuned-full",
    num_train_epochs=1,  # Reduced epochs for full fine-tuning
    per_device_train_batch_size=1,  # Small batch size for memory
    gradient_accumulation_steps=16,  # Larger accumulation for effective batch size of 16
    per_device_eval_batch_size=1,
    warmup_steps=50,
    learning_rate=5e-6,  # Much lower LR for full fine-tuning
    fp16=False,  # Use bf16 instead
    bf16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,  # Keep fewer checkpoints to save disk space
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,  # Essential for memory savings
    dataloader_num_workers=0,  # Reduce to save memory
    report_to=None,  # Disable wandb logging
    max_grad_norm=1.0,  # Gradient clipping for stability
    weight_decay=0.01,  # L2 regularization
    lr_scheduler_type="cosine",  # Better learning rate schedule
    optim="adamw_torch",  # Use standard AdamW
)

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
    pad_to_multiple_of=8  # For efficiency on modern hardware
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Print model parameters info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./llama-math-finetuned-full")
tokenizer.save_pretrained("./llama-math-finetuned-full")

print("Training completed!")