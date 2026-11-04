"""Independent CPU grid/pixel oracles and multimodal mask checks."""

import base64
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import causal_loss_view, mask_sequence  # noqa: E402
from vision import (  # noqa: E402
    build_prompt,
    DEFAULT_PROCESSOR_CONFIG,
    derive_image_geometry,
    load_processor_config,
    model_input_with_completion,
    validate_sampled_prompt,
)


@dataclass
class TextChunk:
    tokens: list[int]

    @property
    def length(self):
        return len(self.tokens)


@dataclass
class InlineImage:
    data: bytes
    format: str
    expected_tokens: int

    @property
    def length(self):
        return self.expected_tokens


@dataclass
class Input:
    chunks: list

    @property
    def length(self):
        return sum(chunk.length for chunk in self.chunks)


CHUNK_TYPES = Input, TextChunk, InlineImage


class TestTokenizer:
    SPECIAL = {"<img>": 100, "<image>": 101, "</img>": 102}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        pieces = re.split(r"(<img>|<image>|</img>)", text)
        return [token for piece in pieces for token in
                ([self.SPECIAL[piece]] if piece in self.SPECIAL
                 else [ord(character) + 256 for character in piece])]

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != {"tokenize": False, "add_generation_prompt": True,
                      "enable_thinking": True}:
            raise AssertionError("Thinking-enabled generation prompt must be used")
        self.messages = messages
        system, user = messages
        content = "".join("<image>" if part["type"] == "input_image" else part["text"]
                          for part in user["content"])
        return ("<|im_start|>system\n" + system["content"] + "<|im_end|>\n"
                "<|im_start|>user\n" + content + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n")


def row_for_size(width, height):
    from PIL import Image

    image = Image.new("RGB", (width, height), (17, 128, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {"responses_create_params": {"input": [
        {"role": "system", "content": "Count stars and return \\boxed{n}."},
        {"role": "user", "content": [
            {"type": "input_image", "image_url": url},
            {"type": "input_text", "text": "How many red stars?"},
        ]},
    ]}}


def scalar_bicubic_rgb(image, target):
    """Independent separable cubic convolution (a=-1/2), no torch calls.

    Pixel-center coordinates use half-pixel scaling. Downsampling widens the
    kernel; cropped edge weights are normalized before the two float passes.
    Only the final result is clamped and ties-to-even rounded to uint8.
    """
    def cubic(distance):
        value = abs(distance)
        if value < 1:
            return (1.5 * value - 2.5) * value * value + 1
        if value < 2:
            return ((-0.5 * value + 2.5) * value - 4) * value + 2
        return 0.0

    def weights(size, count, index):
        scale = size / count
        center = (index + 0.5) * scale - 0.5
        values = [cubic((source - center) / max(scale, 1.0)) for source in range(size)]
        total = sum(values)
        return [value / total for value in values]

    width, height = target
    horizontal = [[sum(image.getpixel((x, y))[channel] * weight
                       for x, weight in enumerate(weights(image.width, width, output_x)))
                   for channel in range(3)] for y in range(image.height) for output_x in range(width)]
    output = []
    for output_y in range(height):
        vertical = weights(image.height, height, output_y)
        for output_x in range(width):
            for channel in range(3):
                value = sum(horizontal[y * width + output_x][channel] * weight
                            for y, weight in enumerate(vertical))
                output.append(round(min(255, max(0, value))))
    return bytes(output)


class GridTests(unittest.TestCase):
    def test_bundled_configuration_identity(self):
        self.assertEqual(DEFAULT_PROCESSOR_CONFIG, ROOT / "processor_config.json")
        self.assertEqual(hashlib.sha256(DEFAULT_PROCESSOR_CONFIG.read_bytes()).hexdigest(),
                         "0b80e5c895052ff2d8e528eca719a6b6e39b7b1b60afb0b94553ffae948e22bc")

    def test_grid_against_fixed_dimension_and_budget_oracles(self):
        # Explicit expectations cover small-image upscaling, rectangular grids,
        # maximum-budget shrinkage and height-before-width budget adjustment.
        cases = [((800, 800), (50, 50), 625), ((997, 997), (64, 64), 1024),
                 ((1024, 1024), (64, 64), 1024), ((1199, 1199), (76, 76), 1444),
                 ((1200, 1200), (76, 76), 1444), ((512, 512), (32, 32), 256),
                 ((256, 256), (32, 32), 256), ((800, 1200), (50, 76), 950),
                 ((1200, 800), (76, 50), 950), ((640, 480), (40, 30), 300),
                 ((1920, 1080), (120, 68), 2040), ((4096, 2160), (158, 84), 3318),
                 ((12000, 12000), (116, 114), 3306), ((500, 3000), (32, 188), 1504),
                 ((1, 1), (32, 32), 256)]
        for (width, height), grid, tokens in cases:
            with self.subTest(size=(width, height)):
                geometry = derive_image_geometry(width, height)
                self.assertEqual((geometry.width_patches, geometry.height_patches), grid)
                self.assertEqual(geometry.expected_tokens, tokens)
                self.assertEqual(geometry.width_patches % 2, 0)
                self.assertEqual(geometry.height_patches % 2, 0)
                self.assertLessEqual(geometry.width_patches * geometry.height_patches, 13312)

    def test_square_token_counts_and_bankers_rounding(self):
        for side, prepared_side, expected in ((800, 800, 625), (997, 1024, 1024),
                                               (1024, 1024, 1024), (1199, 1216, 1444),
                                               (1200, 1216, 1444), (512, 512, 256)):
            with self.subTest(side=side):
                result = derive_image_geometry(side, side)
                self.assertEqual(result.prepared_size, (prepared_side, prepared_side))
                self.assertEqual(result.expected_tokens, expected)

    def test_prepared_grid_is_stable_on_reprocessing(self):
        for width, height in ((997, 997), (1200, 1200), (640, 480), (256, 256), (12000, 12000)):
            geometry = derive_image_geometry(width, height)
            prepared = derive_image_geometry(*geometry.prepared_size)
            self.assertEqual(prepared.prepared_size, geometry.prepared_size)
            self.assertEqual(prepared.expected_tokens, geometry.expected_tokens)

    def test_invalid_or_degenerate_dimensions_fail_closed(self):
        for width, height in ((0, 800), (-1, 800), (True, 800), (1, 1000000)):
            with self.subTest(size=(width, height)), self.assertRaises(ValueError):
                derive_image_geometry(width, height)
        config = load_processor_config()
        config["_downsample_factor"] = 4
        with self.assertRaises(ValueError):
            derive_image_geometry(800, 800, config)


class PromptTests(unittest.TestCase):
    def test_dataset_messages_media_wrappers_and_thinking_prefix(self):
        tokenizer = TestTokenizer()
        row = row_for_size(800, 800)
        prompt, expected, metadata = build_prompt(row, tokenizer, chunk_types=CHUNK_TYPES)
        self.assertEqual(tokenizer.messages, row["responses_create_params"]["input"])
        self.assertIsInstance(prompt.chunks[1], InlineImage)
        self.assertEqual(prompt.chunks[0].tokens[-1], tokenizer.SPECIAL["<img>"])
        self.assertEqual(prompt.chunks[2].tokens[0], tokenizer.SPECIAL["</img>"])
        self.assertEqual(prompt.chunks[1].expected_tokens, 625)
        self.assertEqual(prompt.length, len(expected))
        start, end = metadata["image_target_span"]
        self.assertEqual(expected[start:end], [tokenizer.SPECIAL["<image>"]] * 625)
        self.assertTrue(metadata["enable_thinking"])
        self.assertFalse(metadata["resize_applied"])
        self.assertEqual(metadata["pixel_transform"], "rgb_uint8_preserved")
        self.assertIsNone(metadata["resize_kernel"])
        self.assertEqual(metadata["original_image_sha256"], metadata["prepared_image_sha256"])

    def test_prompt_image_masks_and_first_response_causal_target(self):
        prompt, expected, metadata = build_prompt(row_for_size(800, 800), TestTokenizer(),
                                                   chunk_types=CHUNK_TYPES)
        completion = [42, 43, 44]
        full = model_input_with_completion(prompt, completion, chunk_types=CHUNK_TYPES)
        causal_input = model_input_with_completion(prompt, completion, next_token_inputs=True,
                                                   chunk_types=CHUNK_TYPES)
        self.assertIs(full.chunks[1], prompt.chunks[1])
        self.assertIs(causal_input.chunks[1], prompt.chunks[1])
        self.assertEqual(full.length, len(expected) + 3)
        self.assertEqual(causal_input.length, full.length - 1)
        masked = mask_sequence(expected + completion, response_start=len(expected),
                               advantage=1.5, response_logprobs=[-.1, -.2, -.3])
        start, end = metadata["image_target_span"]
        self.assertEqual(masked.loss_mask[start:end], (0,) * 625)
        self.assertTrue(all(value == 0 for value in masked.advantages[:len(expected)]))
        causal = causal_loss_view(masked)
        self.assertEqual(causal.target_ids[len(expected) - 1], 42)
        self.assertEqual(causal.advantages[len(expected) - 1], 1.5)
        self.assertEqual(causal.old_logprobs[len(expected) - 1], -.1)

    def test_quantized_resize_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "allow_quantized_resize"):
            build_prompt(row_for_size(997, 997), TestTokenizer(), chunk_types=CHUNK_TYPES)

    def test_quantized_resize_uses_exact_grid_and_is_reported(self):
        from PIL import Image

        prompt, expected, metadata = build_prompt(row_for_size(997, 997), TestTokenizer(),
                                                   allow_quantized_resize=True, chunk_types=CHUNK_TYPES)
        self.assertTrue(metadata["resize_applied"])
        self.assertEqual(metadata["pixel_transform"], "bicubic_float_resize_clamp_round_uint8")
        self.assertEqual(metadata["resize_kernel"], "torch_bicubic_align_corners_false_antialias_true")
        self.assertEqual(metadata["expected_image_tokens"], 1024)
        with Image.open(io.BytesIO(prompt.chunks[1].data)) as prepared:
            self.assertEqual(prepared.size, (1024, 1024))
        self.assertEqual(prompt.length, len(expected))

    def test_pixels_against_scalar_antialiased_bicubic_oracle(self):
        from PIL import Image

        source = Image.new("RGB", (9, 7))
        source.putdata([((47*x + 83*y + 7*x*y) % 256,
                         (91*x + 19*y + 11*x*y) % 256,
                         (13*x + 71*y + 5*x*y) % 256) for y in range(7) for x in range(9)])
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        row = row_for_size(9, 7)
        row["responses_create_params"]["input"][1]["content"][0]["image_url"] = (
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
        for budget, target in ((16, (4, 4)), (80, (10, 8))):
            with self.subTest(target=target):
                config = dict(load_processor_config(), patch_size=1,
                              min_num_patches=4 if budget == 16 else 80,
                              max_num_patches=budget)
                prompt, _, metadata = build_prompt(row, TestTokenizer(), processor_config=config,
                    allow_quantized_resize=True, chunk_types=CHUNK_TYPES)
                expected = scalar_bicubic_rgb(source, target)
                with Image.open(io.BytesIO(prompt.chunks[1].data)) as prepared:
                    self.assertEqual(prepared.size, target)
                    self.assertEqual(prepared.tobytes(), expected)
                self.assertEqual(metadata["pixel_transform"], "bicubic_float_resize_clamp_round_uint8")

    def test_rgb_mode_conversion_is_explicit(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("L", (800, 800), 37).save(buffer, format="PNG")
        row = row_for_size(800, 800)
        row["responses_create_params"]["input"][1]["content"][0]["image_url"] = (
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
        prompt, _, metadata = build_prompt(row, TestTokenizer(), chunk_types=CHUNK_TYPES)
        self.assertFalse(metadata["resize_applied"])
        self.assertEqual(metadata["pixel_transform"], "rgb_uint8_conversion")
        with Image.open(io.BytesIO(prompt.chunks[1].data)) as prepared:
            self.assertEqual(prepared.mode, "RGB")
            self.assertEqual(prepared.getpixel((0, 0)), (37, 37, 37))

    def test_context_overflow_and_unsupported_media_fail(self):
        with self.assertRaisesRegex(ValueError, "context"):
            build_prompt(row_for_size(800, 800), TestTokenizer(), max_seq_len=700,
                         max_new_tokens=256, chunk_types=CHUNK_TYPES)
        row = row_for_size(800, 800)
        row["responses_create_params"]["input"][1]["content"][0]["image_url"] = "https://example.com/image.png"
        with self.assertRaisesRegex(ValueError, "inline"):
            build_prompt(row, TestTokenizer(), chunk_types=CHUNK_TYPES)

    def test_sampler_alignment_requires_all_resolved_values(self):
        expected = [7, 101, 101, 9]
        validate_sampled_prompt(SimpleNamespace(prompt_token_ids=list(expected)), expected)
        for returned in (None, [], expected[:-1], [7, 101, 9, 101], [7, True, 101, 9]):
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                validate_sampled_prompt(SimpleNamespace(prompt_token_ids=returned), expected)


if __name__ == "__main__":
    unittest.main()
