import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import os, math, torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
from datasets import load_dataset

class IMDBDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=512):
        self.ds = load_dataset("stanfordnlp/imdb", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[int(idx)]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.long)  # 0/1
        return item

class QwenBackboneIMDBClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels=2, torch_dtype=torch.bfloat16, device_map="auto"):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        hidden = self.lm.config.hidden_size

        # Custom head (example: 2-layer MLP; tweak as you like)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def _pool_last_token(self, last_hidden_state, attention_mask=None):
        # last_hidden_state: [B, T, H]
        if attention_mask is None:
            return last_hidden_state[:, -1, :]
        # index of last non-pad token for each sequence
        idx = attention_mask.long().sum(dim=1) - 1  # [B]
        return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), idx]

    def forward(self, input_ids, attention_mask=None, labels=None):
        # All tensors should be on the same device as model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        pooled = self._pool_last_token(last_hidden, attention_mask)
        logits = self.head(self.dropout(pooled))  # [B, 2]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = os.environ.get("BT_CHECKPOINT_DIR", "outputs/qwen3-imdb-customhead")
MAX_LENGTH = 512

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LR = 2e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
MAX_STEPS = 500
EVAL_STEPS = 100

def get_device(model):
    return next(model.parameters()).device

@torch.no_grad()
def validate(model, loader):
    model.eval()
    device = get_device(model)
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            out = model(**batch)
            total_loss += out["loss"].item()
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    print(f"Val loss: {total_loss/max(1,len(loader)):.4f} | Val acc: {correct/max(1,total):.4f}")
    model.train()

def main():
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("CUDA is required. Please run on a GPU machine.")

    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenBackboneIMDBClassifier(
        model_name=MODEL_NAME,
        num_labels=2,
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    model = model.cuda()

    dataset = IMDBDataset(tokenizer, split="train", max_length=MAX_LENGTH)
    val_size = max(int(0.05 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # If you only want to train the head:
    # for p in model.lm.parameters(): p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = min(MAX_STEPS, steps_per_epoch)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(WARMUP_STEPS, total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0

    for batch_idx, batch in enumerate(train_loader):
        if step >= MAX_STEPS:
            break

        device = get_device(model)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=torch.bfloat16):
            out = model(**batch)
            loss = out["loss"] / GRAD_ACCUM_STEPS

        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            print(f"Step {step} | Loss: {(loss.item() * GRAD_ACCUM_STEPS):.4f}")

            if step > 0 and step % EVAL_STEPS == 0:
                validate(model, val_loader)

            step += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Save custom head + base LM weights together
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
