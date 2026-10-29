import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock, patch

from fixtures import PROCESSOR_CONFIG
from PIL import Image

import data
import train
from vision import _prepare_image, derive_image_geometry, load_processor_config


class Tokenizer:
    specials: ClassVar[dict[str, int]] = {
        "<image>": 1001,
        "<img>": 1002,
        "</img>": 1003,
        "<im_end>": 1004,
    }

    def apply_chat_template(self, messages, **kwargs):
        answer = messages[-1]["content"][0]["text"]
        return (
            "user:<image>Describe this image.\nassistant:<think>\n\n</think>\n"
            + answer
            + "<im_end>"
        )

    def __call__(self, text, **kwargs):
        tokens, offsets, index = [], [], 0
        while index < len(text):
            special = next(
                (item for item in self.specials if text.startswith(item, index)), None
            )
            width = len(special) if special else 1
            tokens.append(self.specials[special] if special else ord(text[index]))
            offsets.append((index, index + width))
            index += width
        return {"input_ids": tokens, "offset_mapping": offsets}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]


class DataTests(unittest.TestCase):
    def test_exact_receipt_serialization(self):
        value = {
            "menu": [{"price": "12", "nm": "Tea"}, {"nm": "Rice"}],
            "total": {"total_price": "20"},
        }
        self.assertEqual(
            data.json2token(value),
            "<s_total><s_total_price>20</s_total_price></s_total>"
            "<s_menu><s_price>12</s_price><s_nm>Tea</s_nm><sep/><s_nm>Rice</s_nm></s_menu>",
        )
        self.assertEqual(data.json2token({"text_sequence": "literal"}), "literal")

    def test_multimodal_next_token_alignment(self):
        answer = "<s_total>42</s_total>"
        row = data.encode_row(Tokenizer(), answer, 256, 4096)
        self.assertEqual(
            len(row["target_tokens"]),
            len(row["prefix_ids"]) + 256 + len(row["suffix_ids"]),
        )
        self.assertEqual(row["active_tokens"], len(answer) + 1)
        self.assertEqual(row["target_tokens"][-1], 1004)
        first = next(
            index for index, value in enumerate(row["target_tokens"]) if value != -100
        )
        self.assertEqual(row["target_tokens"][first], ord("<"))
        media_start = len(row["prefix_ids"])
        self.assertTrue(
            all(
                value == -100
                for value in row["target_tokens"][media_start : media_start + 256]
            )
        )

    def test_overlong_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            data.encode_row(Tokenizer(), "answer", 256, 16)

    def test_larger_context_preserves_entire_answer(self):
        answer = "answer" * 700
        row = data.encode_row(Tokenizer(), answer, 256, 8192)
        self.assertGreater(row["sequence_tokens"], 4096)
        self.assertEqual(row["active_tokens"], len(answer) + 1)

    def test_datum_keeps_image_in_place(self):
        row = data.encode_row(Tokenizer(), "answer", 256, 4096)
        row.update(image="receipt.png", image_format="png")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "data" / "prepared" / "receipt.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image bytes")
            with patch.object(data, "ROOT", root):
                encoded = data.datum(row)
        self.assertEqual(
            [chunk.type for chunk in encoded.model_input.chunks],
            ["encoded_text", "image", "encoded_text"],
        )
        self.assertEqual(encoded.model_input.length, len(row["target_tokens"]))
        self.assertEqual(encoded.model_input.chunks[1].data, b"image bytes")


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.processor_path = Path(self.directory.name) / "preprocessor_config.json"
        self.processor_path.write_text(json.dumps(PROCESSOR_CONFIG))

    def test_large_receipt_is_checked_after_reference_resize(self):
        buffer = io.BytesIO()
        Image.new("RGB", (1024, 7000), "white").save(buffer, format="PNG")
        _, _, geometry, _ = _prepare_image(
            buffer.getvalue(), "png", load_processor_config(self.processor_path), True
        )
        width, height = geometry.prepared_size
        self.assertLess(width * height * 3, 18 * 1024 * 1024)

    def test_divisible_and_stable_grid(self):
        config = load_processor_config(self.processor_path)
        for width, height in [(256, 256), (864, 1296), (1920, 1080), (720, 3000)]:
            geometry = derive_image_geometry(width, height, config)
            self.assertEqual(geometry.width_patches % 2, 0)
            self.assertEqual(geometry.height_patches % 2, 0)
            repeated = derive_image_geometry(*geometry.prepared_size, config)
            self.assertEqual(repeated.prepared_size, geometry.prepared_size)
            self.assertEqual(repeated.expected_tokens, geometry.expected_tokens)

    def test_resize_requires_explicit_opt_in(self):
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), "white").save(buffer, format="PNG")
        with self.assertRaisesRegex(ValueError, "Explicitly enable"):
            _prepare_image(
                buffer.getvalue(),
                "png",
                load_processor_config(self.processor_path),
                False,
            )
        prepared, image_format, geometry, audit = _prepare_image(
            buffer.getvalue(), "png", load_processor_config(self.processor_path), True
        )
        with Image.open(io.BytesIO(prepared)) as image:
            self.assertEqual(image.size, geometry.prepared_size)
        self.assertEqual(image_format, "png")
        self.assertFalse(audit["exact_image_pixels"])


class TrainingTests(unittest.TestCase):
    def test_validation_weights_active_tokens(self):
        client = Mock()
        client.forward.side_effect = [
            Mock(result=Mock(return_value=Mock(loss=2.0))),
            Mock(result=Mock(return_value=Mock(loss=8.0))),
        ]
        rows = [{"active_tokens": 10}, {"active_tokens": 30}]
        with patch.object(train, "datum", side_effect=lambda row: row):
            value = train.validation(client, rows, 1, 60)
        self.assertEqual(value, 6.5)
        self.assertEqual(client.forward.call_count, 2)
        self.assertFalse(client.forward_backward.called)
        self.assertFalse(client.optim_step.called)

    def test_shuffle_is_reproducible_and_each_epoch_complete(self):
        examples = list(range(16))
        first = train.batches(examples, 8, 42)
        second = train.batches(examples, 8, 42)
        for _ in range(3):
            epoch = next(first) + next(first)
            repeated = next(second) + next(second)
            self.assertEqual(epoch, repeated)
            self.assertEqual(sorted(epoch), examples)

    def test_failed_optimizer_updates_are_rejected(self):
        for metrics in [{"grad_norm": math.nan}, {"skipped_iterations": 1}]:
            with self.assertRaises(RuntimeError):
                train.check_optimizer_metrics(metrics)

    def test_three_update_lifecycle_cleans_up_only_owned_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "steps": 400,
                "seed": 42,
                "batch_size": 1,
                "base_model": "test/model",
                "rank": 64,
                "max_seq_len": 4096,
                "learning_rate": 0.0002,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
                "weight_decay": 0.01,
                "grad_clip_norm": 1.0,
                "validation_interval": 100,
                "checkpoint_interval": 400,
                "wandb_prefix": "baseten_",
                "wandb_entity": "test",
                "wandb_project": "test",
                "team": "test",
                "replicas": 1,
            }
            client = Mock(run_id="owned-run")
            client.forward.return_value.result.return_value = Mock(loss=0.7, metrics={})
            client.forward_backward.return_value.result.return_value = Mock(
                loss=0.5, metrics={}
            )
            client.optim_step.return_value.result.return_value = Mock(metrics={})
            client.save_state.return_value.result.return_value = Mock(
                path="bt://owned/trainer"
            )
            client.save_weights_for_sampler.return_value.result.return_value = Mock(
                path="bt://owned/adapter"
            )
            checkpoint = Mock(checkpoint_id="step-3")
            checkpoint.model_dump.return_value = {"checkpoint_id": "step-3"}
            client.list_checkpoints.return_value = [checkpoint]
            service = Mock()
            service.create_lora_training_client.return_value = client
            run = Mock(url="https://wandb.test/run", summary={})
            rows = [{"active_tokens": 1}, {"active_tokens": 1}]
            with (
                patch.multiple(
                    train,
                    ROOT=root,
                    settings=Mock(return_value=config),
                    load_env=Mock(),
                    validate_catalog=Mock(return_value={}),
                    prepare=Mock(
                        return_value=({"train": rows, "validation": rows}, {})
                    ),
                    datum=Mock(side_effect=lambda row: row),
                    validation=Mock(return_value=0.6),
                    ServiceClient=Mock(return_value=service),
                    api_request=Mock(),
                ),
                patch.object(train.wandb, "init", return_value=run),
                patch.object(train.wandb, "Image", return_value="plot"),
                patch("plot_loss.plot", return_value=root / "plot.png"),
                patch.dict("os.environ", {"WANDB_API_KEY": "fake-key"}, clear=True),
                patch(
                    "sys.argv",
                    [
                        "train.py",
                        "--steps",
                        "3",
                        "--name",
                        "baseten_test",
                    ],
                ),
            ):
                train.main()
                train.api_request.assert_called_once_with(
                    "/v1/loops/runs/owned-run/deactivate", method="POST"
                )
                self.assertEqual(client.forward_backward.call_count, 3)
                self.assertEqual(client.optim_step.call_count, 3)
                client.close.assert_called_once()
                service.create_lora_training_client.assert_called_once()
                self.assertFalse(
                    service.create_lora_training_client.call_args.kwargs["with_sampler"]
                )
            completed = json.loads(
                (root / "outputs" / "baseten_test" / "completion.json").read_text()
            )
            self.assertTrue(completed["complete"])
            self.assertEqual(completed["updates"], 3)


if __name__ == "__main__":
    unittest.main()
