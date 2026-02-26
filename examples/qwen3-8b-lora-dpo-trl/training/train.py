"""
DPO Training Script for Qwen2.5-8B with LoRA
Uses TRL's DPOTrainer on HumanLLMs/Human-Like-DPO-Dataset
"""

import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import os


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training Script for Qwen2.5-8B with LoRA")
    
    # Model and dataset arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="Pretrained model name or path")
    parser.add_argument("--dataset_name", type=str, default="HumanLLMs/Human-Like-DPO-Dataset",
                        help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default=os.getenv("BT_CHECKPOINT_DIR", "./dpo_qwen_output"),
                        help="Output directory for model checkpoints")
    parser.add_argument("--logging_dir", type=str, default="./logs",
                        help="Directory for logging")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs='+',
                    default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                    help="Target modules for LoRA")
    parser.add_argument("--lora_bias", type=str, default="none",
                        help="LoRA bias type")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    
    # DPO specific arguments (matching DPOConfig)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient for DPO")
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        help="DPO loss type")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=256,
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Maximum completion length")
    parser.add_argument("--label_pad_token_id", type=int, default=-100,
                        help="Label padding token ID")
    parser.add_argument("--truncation_mode", type=str, default="keep_end",
                        help="Truncation mode for sequences")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor")
    parser.add_argument("--disable_dropout", action="store_true", default=True,
                        help="Disable dropout during training")
    
    # Other arguments
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision")
    parser.add_argument("--no_bf16", action="store_false", dest="bf16",
                        help="Don't use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", 
                        dest="gradient_checkpointing",
                        help="Don't use gradient checkpointing")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Test set size for train/test split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # LoRA Configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # DPO Training Configuration
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        use_cache=False,
    )
    
    # Prepare model for training
    model.config.use_cache = False
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)
    
    # Split dataset
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=args.test_size, seed=args.seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Dataset columns: {train_dataset.column_names}")
    
    # Initialize DPO Trainer
    print("Initializing DPO Trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )
    
    # Train
    print("Starting training...")
    dpo_trainer.train()
    
    # Save final model
    print("Saving final model...")
    final_model_path = os.path.join(args.output_dir, "final_model")
    dpo_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("Training complete!")
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()