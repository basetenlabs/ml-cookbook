import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from evaluate_f1 import generate


class GenerationTests(unittest.TestCase):
    def examples(self, count):
        return [
            {
                "index": index,
                "prompt": "fixture",
                "expected_prompt_ids": [42],
                "ground_truth": {"nm": "Coffee"},
                "prompt_tokens": 1,
                "original_image_sha256": f"fixture-{index}",
            }
            for index in range(count)
        ]

    def sampler(self, prompt_tokens):
        sequence = SimpleNamespace(tokens=[123], stop_reason="stop")
        response = SimpleNamespace(
            sequences=[sequence],
            prompt_token_ids=prompt_tokens,
            model_dump=lambda **kwargs: {
                "sequences": [{"tokens": [123], "stop_reason": "stop"}]
            },
        )
        return SimpleNamespace(sample_async=AsyncMock(return_value=response))

    def test_complete_generation_requires_all_100_distinct_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sampler = self.sampler([42])
            tokenizer = Mock()
            tokenizer.decode.return_value = "<s_nm>Coffee</s_nm>"
            rows = asyncio.run(
                generate(sampler, self.examples(100), tokenizer, output, 10, 128)
            )
            self.assertEqual(len(rows), 100)
            self.assertEqual({row["index"] for row in rows}, set(range(100)))
            self.assertEqual(sampler.sample_async.await_count, 100)
            saved = [
                json.loads(line)
                for line in (output / "rollouts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(saved), 100)

    def test_partial_coverage_is_not_a_final_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = Mock()
            tokenizer.decode.return_value = "<s_nm>Coffee</s_nm>"
            with self.assertRaisesRegex(ValueError, "all 100 distinct"):
                asyncio.run(
                    generate(
                        self.sampler([42]),
                        self.examples(99),
                        tokenizer,
                        Path(temporary),
                        10,
                        128,
                    )
                )

    def test_sampler_prompt_must_match_without_truncation(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "complete, untruncated"),
        ):
            asyncio.run(
                generate(
                    self.sampler([99]),
                    self.examples(1),
                    Mock(),
                    Path(temporary),
                    10,
                    128,
                )
            )


if __name__ == "__main__":
    unittest.main()
