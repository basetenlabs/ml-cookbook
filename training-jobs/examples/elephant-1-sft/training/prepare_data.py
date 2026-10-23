"""Write BANKING77 as decision-model rows: one multiple-choice question per message, one-hot target.

Replace this with your own data. Each row needs a `state` (string or JSON), a `question` (`choice` with
`criteria` {label: description or null},`noul` for yes/no, or `score` with a list of levels) and a `target`
{label: probability}; soft targets are fine.
"""

import json
import sys

from datasets import load_dataset

out_dir = sys.argv[1]
for split in ("train", "test"):
    ds = load_dataset("mteb/banking77", split=split)
    question = {
        "type": "choice",
        "instructions": "Which intent does the customer message express?",
        "criteria": dict.fromkeys(sorted(set(ds["label_text"]))),
    }
    with open(f"{out_dir}/{split}.jsonl", "w") as f:
        for i, ex in enumerate(ds):
            row = {
                "id": f"{split}/{i}",
                "state": ex["text"],
                "question": question,
                "target": {ex["label_text"]: 1.0},
            }
            f.write(json.dumps(row) + "\n")
