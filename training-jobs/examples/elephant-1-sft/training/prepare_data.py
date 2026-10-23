"""Script writes public BANKING77 dataset as rows for decision model training.

Each row contains a multiple-choice question per message, with one-hot target.
"""

import json
import sys

from datasets import load_dataset

out_dir = sys.argv[1]
ds = load_dataset("mteb/banking77")
question = {
    "type": "choice",
    "instructions": "Which intent does the customer message express?",
    "criteria": dict.fromkeys(sorted(set(ds["train"]["label_text"]))),
}
for split in ("train", "test"):
    with open(f"{out_dir}/{split}.jsonl", "w") as f:
        for i, ex in enumerate(ds[split]):
            row = {
                "id": f"{split}/{i}",
                "state": ex["text"],
                "question": question,
                "target": {ex["label_text"]: 1.0},
            }
            f.write(json.dumps(row) + "\n")
