"""Fine-tune Qwen3-0.6B using BDN-mounted weights.

Model weights are pre-loaded at /app/models/Qwen/Qwen3-0.6B via BDN.
No download code needed — just load from the mount path.
"""

import os
import math
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

# BDN mount path — weights are already here when the container starts.
MODEL_PATH = "/app/models/Qwen/Qwen3-0.6B"
OUTPUT_DIR = os.environ.get("BT_CHECKPOINT_DIR", "/tmp/training_checkpoints")

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1
NUM_EPOCHS = 1
MAX_STEPS = 20
MAX_LENGTH = 2048
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 5
EVAL_STEPS = 10
VAL_SET_RATIO = 0.05
SAVE_EVERY_STEPS = 10


def messages_to_text(messages):
    chunks = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        chunks.append(f"{role}: {content}")
    return "\n".join(chunks)


class ChatDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=2048):
        ds = load_dataset("winglian/pirate-ultrachat-10k", split=split)
        ds = ds.filter(
            lambda x: x.get("messages") is not None and len(x["messages"]) > 0
        )
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[int(idx)]
        text = messages_to_text(ex["messages"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_device(model):
    return next(model.parameters()).device


def validate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_count = 0
    device = get_device(model)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            total_loss += outputs.loss.item()
            total_count += 1
    avg_loss = total_loss / max(total_count, 1)
    print(f"Validation loss: {avg_loss:.4f}")
    model.train()


def main():
    use_cuda = torch.cuda.is_available()

    # Verify BDN mount
    print(f"Loading model from BDN mount: {MODEL_PATH}")
    assert os.path.exists(MODEL_PATH), f"BDN mount not found at {MODEL_PATH}"
    print(f"  Files: {os.listdir(MODEL_PATH)}")

    # Load from the BDN mount path — no download happens here.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if use_cuda else None,
    )

    # Dataset
    full_dataset = ChatDataset(tokenizer, max_length=MAX_LENGTH)
    val_size = max(int(len(full_dataset) * VAL_SET_RATIO), 1)
    train_size = len(full_dataset) - val_size
    print(f"Dataset: {len(full_dataset)} total | {train_size} train | {val_size} val")

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = min(MAX_STEPS, steps_per_epoch * NUM_EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(WARMUP_STEPS, total_steps),
        num_training_steps=total_steps,
    )

    # Train
    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(train_loader):
            if step >= MAX_STEPS:
                break

            device = get_device(model)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_cuda, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS

            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                print(f"Step {step} | Loss: {(loss.item() * GRAD_ACCUM_STEPS):.4f}")

                if step > 0 and step % EVAL_STEPS == 0:
                    validate(model, val_loader)

                if step > 0 and step % SAVE_EVERY_STEPS == 0:
                    save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")

                step += 1

        if step >= MAX_STEPS:
            break

    # Final save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
