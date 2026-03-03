import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from torch.utils.data import random_split

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = os.environ.get("BT_CHECKPOINT_DIR", "outputs/qwen3-imdb-hf-trainer")
MAX_LENGTH = 512
NUM_LABELS = 2

# Training hyperparameters
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LR = 2e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
MAX_STEPS = 500
EVAL_STEPS = 100
SAVE_STEPS = 100
LOGGING_STEPS = 10


def tokenize_function(examples, tokenizer, max_length=MAX_LENGTH):
    """Tokenize the text examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # We'll use DataCollatorWithPadding
        max_length=max_length,
    )


def prepare_dataset(tokenizer, max_length=MAX_LENGTH):
    """Load and prepare the IMDB dataset."""
    # Load the dataset
    dataset = load_dataset("stanfordnlp/imdb")
    
    # Tokenize the datasets
    tokenized_train = dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_test = dataset["test"].map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    
    # Rename label column to labels (required by Trainer)
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")
    
    # Use a small validation split from training data
    train_size = int(0.95 * len(tokenized_train))
    val_size = len(tokenized_train) - train_size
    train_val_split = tokenized_train.train_test_split(
        test_size=val_size / len(tokenized_train),
        seed=42
    )
    
    return train_val_split["train"], train_val_split["test"], tokenized_test


def main():
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("CUDA is required. Please run on a GPU machine.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with sequence classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Set pad_token_id in model config to match tokenizer
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_dataset(tokenizer, MAX_LENGTH)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  # We'll use max_steps instead
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,  # Use bfloat16 via model dtype instead
        bf16=True,
        dataloader_pin_memory=True,
        save_total_limit=2,
    )

    # Compute metrics function
    def compute_metrics(eval_pred):
        import numpy as np
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test accuracy: {test_results.get('eval_accuracy', 0):.4f}")

    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

