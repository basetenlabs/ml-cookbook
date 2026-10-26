"""Script writes public BANKING77 dataset as rows for decision model training.

Each row contains a multiple-choice question per message, with one-hot target.
"""

import json
import sys
from typing import Literal, NotRequired, TypedDict

from datasets import load_dataset


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: NotRequired[str]
    criteria: dict[str, str | None]  # label -> description


class NoulQuestion(TypedDict):
    type: Literal["noul"]  # yes/no; target labels are "yes" and "no"
    instructions: NotRequired[str]
    criteria: NotRequired[dict[Literal["true", "false"], str]]  # descriptions


class ScoreQuestion(TypedDict):
    type: Literal["score"]  # target labels are "0" to str(len(criteria) - 1)
    instructions: NotRequired[str]
    criteria: list[str]  # one description per level


class Row(TypedDict):
    id: str
    state: str | dict  # text, or JSON
    question: ChoiceQuestion | NoulQuestion | ScoreQuestion
    target: dict[str, float]  # label -> probability; soft targets are fine


out_dir = sys.argv[1]
ds = load_dataset("mteb/banking77")
question: ChoiceQuestion = {
    "type": "choice",
    "instructions": "Which intent does the customer message express?",
    "criteria": dict.fromkeys(sorted(set(ds["train"]["label_text"]))),
}
for split in ("train", "test"):
    with open(f"{out_dir}/{split}.jsonl", "w") as f:
        for i, ex in enumerate(ds[split]):
            row: Row = {
                "id": f"{split}/{i}",
                "state": ex["text"],
                "question": question,
                "target": {ex["label_text"]: 1.0},
            }
            f.write(json.dumps(row) + "\n")
