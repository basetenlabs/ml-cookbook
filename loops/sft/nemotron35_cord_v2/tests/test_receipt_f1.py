import unittest

from data import json2token
from evaluate_f1 import sampler_source
from receipt_f1 import parse_prediction, score


class ReceiptF1Tests(unittest.TestCase):
    def test_baseline_sampler_has_no_adapter_path(self):
        self.assertEqual(
            sampler_source("base-model", None), {"base_model": "base-model"}
        )

    def test_checkpoint_sampler_pins_only_the_checkpoint(self):
        path = "bt://loops:owned-run/sampler_weights/step-400"
        self.assertEqual(sampler_source("base-model", path), {"model_path": path})

    def test_nested_repeated_fields_are_counted_as_a_multiset(self):
        receipt = {
            "menu": [{"nm": "Coffee", "price": "10"}, {"nm": "Coffee", "price": "10"}],
            "total": {"price": "20"},
        }
        result = score(
            [
                {
                    "ground_truth": receipt,
                    "text": json2token(receipt),
                    "finish_reason": "stop",
                }
            ]
        )
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["true_positive"], 5)

    def test_reasoning_opened_in_prompt_is_not_part_of_prediction(self):
        self.assertEqual(
            parse_prediction(
                "Reasoning from prompt-opened think tag.</think>\n<s_nm>Coffee</s_nm>"
            ),
            {"nm": "Coffee"},
        )

    def test_invalid_and_truncated_receipt_counts_missing_fields(self):
        rows = [
            {
                "ground_truth": {"nm": "Coffee"},
                "text": "<s_nm>Coffee",
                "finish_reason": "length",
            }
        ]
        result = score(rows)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["ground_truth_fields"], 1)
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertEqual(result["output_truncations"], 1)

    def test_wrong_value_and_extra_field_reduce_precision_and_recall(self):
        rows = [
            {
                "ground_truth": {"nm": "Coffee", "price": "10"},
                "text": '{"nm":"Coffee","price":"11","extra":"x"}',
                "finish_reason": "stop",
            }
        ]
        result = score(rows)
        self.assertEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 1 / 2)
        self.assertEqual(result["f1"], 2 / 5)

    def test_unfinished_thinking_and_duplicate_keys_fail(self):
        for text in ("<think>unfinished", "<s_nm>A</s_nm><s_nm>B</s_nm>"):
            with self.assertRaises(ValueError):
                parse_prediction(text)


if __name__ == "__main__":
    unittest.main()
