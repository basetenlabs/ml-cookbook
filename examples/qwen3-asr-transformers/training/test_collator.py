"""CPU regression checks for label masking, without loading model weights."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
import unittest

import torch


def load_collator():
    # Exercise the production class without importing GPU/model dependencies.
    path = Path(__file__).with_name("qwen3_asr_sft.py")
    tree = ast.parse(path.read_text())
    collator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DataCollatorForQwen3ASRFinetuning"
    )
    namespace = dict(
        dataclass=dataclass,
        Any=Any,
        Dict=Dict,
        List=List,
        torch=torch,
        load_audio=lambda path, sr: path,
    )
    exec(
        compile(ast.Module(body=[collator], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[collator.name]


class Processor:
    """Token-ID fixture with Qwen's left-padding default and distinct PAD/EOS."""

    tokenizer = SimpleNamespace(eos_token=" 99", pad_token_id=0)

    def __call__(
        self, text, audio, return_tensors, padding, truncation, padding_side="left"
    ):
        rows = [[int(token) for token in value.split()] for value in text]
        width = max(map(len, rows))
        rows = [
            row + [0] * (width - len(row))
            if padding_side == "right"
            else [0] * (width - len(row)) + row
            for row in rows
        ]
        ids = torch.tensor(rows)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class CollatorTest(unittest.TestCase):
    def test_only_reference_and_eos_tokens_are_supervised(self):
        short = {"audio": "short", "prefix_text": "10 11 ", "target": "20"}
        long = {"audio": "long", "prefix_text": "10 12 13 ", "target": "21 22 23 24 25"}
        # Include padding shorter and longer than the prompt, plus a singleton.
        nearby = {"audio": "nearby", "prefix_text": "10 11 ", "target": "21 22"}
        for features in ([short], [short, nearby], [short, long], [long, short]):
            with self.subTest(targets=[f["target"] for f in features]):
                batch = load_collator()(Processor())(features)
                for index, feature in enumerate(features):
                    labels = batch["labels"][index]
                    expected = [int(token) for token in feature["target"].split()] + [
                        99
                    ]
                    self.assertEqual(labels[labels != -100].tolist(), expected)
                    attention = batch["attention_mask"][index]
                    self.assertTrue(torch.all(labels[attention == 0] == -100))


if __name__ == "__main__":
    unittest.main()
