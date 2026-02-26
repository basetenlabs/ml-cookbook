import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

OUTPUT_DIR = os.environ.get("BT_CHECKPOINT_DIR", "outputs/qwen3-0.6b")
MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():
    # Load model — 0.6B fits easily on H200 without quantization
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("winglian/pirate-ultrachat-10k", split="train")

    # SFT config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_steps=20,
        save_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        logging_steps=1,
        max_length=2048,
        packing=True,
        report_to="none",
    )

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    print("Training complete!")


if __name__ == "__main__":
    main()
