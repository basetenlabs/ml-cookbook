"""Nemotron image prompts with explicit resizing and audited token positions.

The bundled processor_config.json defines the dynamic image grid. Resizing
uses float bicubic interpolation, followed by clamp/round to uint8 for PNG
transport. This quantization requires opt-in; sampler IDs are checked exactly.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any


DEFAULT_PROCESSOR_CONFIG = Path(__file__).resolve().parent / "processor_config.json"
IMAGE_TOKEN = "<image>"
IMAGE_START = "<img>"
IMAGE_END = "</img>"


def load_processor_config(path: str | Path = DEFAULT_PROCESSOR_CONFIG) -> dict:
    config = json.loads(Path(path).read_text())
    if "image_processor" in config:
        config = config["image_processor"]
    _validate_processor_config(config)
    return config


def _validate_processor_config(config: dict) -> None:
    required = {"patch_size", "downsample_ratio", "min_num_patches",
                "max_num_patches", "max_model_len"}
    if not required.issubset(config):
        raise ValueError("Nemotron image processor configuration is incomplete")
    for key in ("patch_size", "min_num_patches", "max_num_patches", "max_model_len"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid Nemotron processor setting: {key}")
    ratio = config["downsample_ratio"]
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid Nemotron downsample ratio")
    factor = int(round(1.0 / ratio))
    if factor < 1 or not math.isclose(factor * ratio, 1.0, abs_tol=1e-12):
        raise ValueError("Nemotron pixel-shuffle reduction must be an integer")
    if config.get("_downsample_factor", factor) != factor:
        raise ValueError("Conflicting Nemotron pixel-shuffle settings")
    if config["min_num_patches"] > config["max_num_patches"]:
        raise ValueError("Nemotron minimum patch budget exceeds its maximum")


@dataclass(frozen=True)
class ImageGeometry:
    original_width: int
    original_height: int
    width_patches: int
    height_patches: int
    patch_size: int
    downsample_factor: int
    patch_budget: int

    @property
    def prepared_size(self) -> tuple[int, int]:
        return self.width_patches * self.patch_size, self.height_patches * self.patch_size

    @property
    def expected_tokens(self) -> int:
        return self.width_patches * self.height_patches // self.downsample_factor ** 2

    @property
    def needs_resize(self) -> bool:
        return self.prepared_size != (self.original_width, self.original_height)


def derive_image_geometry(width: int, height: int, config: dict | None = None) -> ImageGeometry:
    """Calculate the single-image dynamic patch grid from bundled settings.

    Python's round(x + .5), including ties-to-even, is intentional.
    """
    config = load_processor_config() if config is None else config
    _validate_processor_config(config)
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    patch = config["patch_size"]
    divisor = int(round(1.0 / config["downsample_ratio"]))
    minimum = config["min_num_patches"]
    budget = max((config["max_model_len"] - 4) * divisor ** 2, minimum)
    budget = max(min(budget, config["max_num_patches"]), minimum)
    closest_h = round(height / patch + 0.5)
    closest_w = round(width / patch + 0.5)
    if closest_h <= 0 or closest_w <= 0:
        raise ValueError("Image is too small for the configured patch rule")
    factor = min(math.sqrt(budget / (closest_h * closest_w)), 1.0)
    target_h = math.floor(factor * closest_h)
    target_w = math.floor(factor * closest_w)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Image aspect ratio produces an empty patch grid")
    if budget > minimum and target_h * target_w < minimum:
        up = math.sqrt(minimum / (target_h * target_w))
        target_h = math.ceil(up * target_h)
        target_w = math.ceil(up * target_w)
    rem_h = target_h % divisor
    if rem_h:
        inc_h = divisor - rem_h
        target_h = (target_h + inc_h if (target_h + inc_h) * target_w <= budget
                    else max(divisor, target_h - rem_h))
    rem_w = target_w % divisor
    if rem_w:
        inc_w = divisor - rem_w
        target_w = (target_w + inc_w if target_h * (target_w + inc_w) <= budget
                    else max(divisor, target_w - rem_w))
    if target_h * target_w > budget or min(target_h, target_w) < divisor:
        raise ValueError("Image grid violates the available patch budget")
    return ImageGeometry(width, height, target_w, target_h, patch, divisor, budget)


def _prepare_image(raw: bytes, declared_format: str, config: dict,
                   allow_quantized_resize: bool) -> tuple[bytes, str, ImageGeometry, dict]:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as opened:
        actual_format = {"PNG": "png", "JPEG": "jpeg"}.get(opened.format)
        if actual_format != declared_format:
            raise ValueError("Inline image MIME type does not match its decoded format")
        width, height = opened.size
        if width * height > 50_000_000 or width * height * 3 > 18 * 1024 * 1024:
            raise ValueError("Decoded RGB image exceeds the Loops image limit")
        geometry = derive_image_geometry(width, height, config)
        original_mode = opened.mode
        image = opened.convert("RGB")
        image.load()
    if geometry.needs_resize and not allow_quantized_resize:
        raise ValueError(
            f"Image {width}x{height} requires resize to {geometry.prepared_size}; "
            "float interpolation is clamped/rounded to uint8 for PNG transport. "
            "Explicitly enable allow_quantized_resize."
        )
    if geometry.needs_resize:
        # Interpolate in float; PNG transport then requires uint8 quantization.
        import numpy as np
        import torch

        array = np.asarray(image, dtype=np.uint8).copy()
        pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        target_w, target_h = geometry.prepared_size
        pixels = torch.nn.functional.interpolate(
            pixels, size=(target_h, target_w), mode="bicubic",
            align_corners=False, antialias=True,
        )
        quantized = pixels.clamp(0, 255).round().to(torch.uint8)
        image = Image.fromarray(quantized.squeeze(0).permute(1, 2, 0).numpy())
    if not geometry.needs_resize and original_mode == "RGB":
        prepared, image_format = raw, declared_format
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        prepared, image_format = buffer.getvalue(), "png"
    stable = derive_image_geometry(*geometry.prepared_size, config)
    if stable.prepared_size != geometry.prepared_size or stable.expected_tokens != geometry.expected_tokens:
        raise ValueError("Prepared image would change grid when processed again")
    audit = {
        "original_size": [width, height],
        "prepared_size": list(geometry.prepared_size),
        "original_image_sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_image_sha256": hashlib.sha256(prepared).hexdigest(),
        "image_format": image_format,
        "expected_image_tokens": geometry.expected_tokens,
        "patch_grid": [geometry.width_patches, geometry.height_patches],
        "resize_applied": geometry.needs_resize,
        "pixel_transform": ("bicubic_float_resize_clamp_round_uint8" if geometry.needs_resize
                            else "rgb_uint8_preserved" if original_mode == "RGB"
                            else "rgb_uint8_conversion"),
        "resize_kernel": ("torch_bicubic_align_corners_false_antialias_true"
                          if geometry.needs_resize else None),
    }
    return prepared, image_format, geometry, audit


def _chunk_types(chunk_types=None):
    if chunk_types is not None:
        return chunk_types
    from baseten.loops import ModelInput, EncodedTextChunk, ImageChunk
    return ModelInput, EncodedTextChunk, ImageChunk


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Tokenizer returned invalid token IDs")
    return ids


def build_prompt(row: dict, tokenizer: Any, *, processor_config: dict | str | Path | None = None,
                 max_seq_len: int = 4096, max_new_tokens: int = 256,
                 allow_quantized_resize: bool = False, chunk_types=None) -> tuple[Any, list[int], dict]:
    """Return (multimodal ModelInput, expected sampler prompt IDs, audit).

    The dataset's system/user text and content ordering are retained. Exactly
    one inline image is required, matching this frozen star-count dataset.
    """
    if type(max_seq_len) is not int or type(max_new_tokens) is not int or min(max_seq_len, max_new_tokens) <= 0:
        raise ValueError("Context and completion budgets must be positive integers")
    if type(allow_quantized_resize) is not bool:
        raise ValueError("allow_quantized_resize must be an explicit boolean")
    config = (load_processor_config(processor_config) if isinstance(processor_config, (str, Path))
              else load_processor_config() if processor_config is None else processor_config)
    _validate_processor_config(config)
    messages = row.get("responses_create_params", {}).get("input")
    if not isinstance(messages, list) or [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("Expected dataset system and user messages")
    images = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "input_image":
                    images.append(part)
                elif part.get("type") != "input_text" or not isinstance(part.get("text"), str):
                    raise ValueError("Unsupported dataset message content")
        elif not isinstance(content, str):
            raise ValueError("Unsupported dataset message content")
    if len(images) != 1:
        raise ValueError("Expected exactly one inline dataset image")
    url = images[0].get("image_url")
    match = re.fullmatch(r"data:image/(png|jpeg);base64,([A-Za-z0-9+/=]+)", url or "")
    if match is None:
        raise ValueError("Expected an inline PNG or JPEG data URL")
    try:
        raw = base64.b64decode(match[2], validate=True)
    except ValueError as error:
        raise ValueError("Invalid inline image base64") from error
    prepared, image_format, geometry, audit = _prepare_image(
        raw, match[1], config, allow_quantized_resize)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if not isinstance(rendered, str) or rendered.count(IMAGE_TOKEN) != 1:
        raise ValueError("Chat template did not produce exactly one image placeholder")
    if IMAGE_START in rendered or IMAGE_END in rendered:
        raise ValueError("Chat template already wrapped the image; processor expansion is ambiguous")
    image_id = _token_ids(tokenizer, IMAGE_TOKEN)
    if len(image_id) != 1:
        raise ValueError("The managed tokenizer lacks a single Nemotron image token")
    prefix, suffix = rendered.split(IMAGE_TOKEN)
    prefix_ids = _token_ids(tokenizer, prefix + IMAGE_START)
    suffix_ids = _token_ids(tokenizer, IMAGE_END + suffix)
    expected = prefix_ids + image_id * geometry.expected_tokens + suffix_ids
    expanded = prefix + IMAGE_START + IMAGE_TOKEN * geometry.expected_tokens + IMAGE_END + suffix
    if _token_ids(tokenizer, expanded) != expected:
        raise ValueError("Chunk boundaries do not match full processor tokenization")
    if len(expected) + max_new_tokens > max_seq_len:
        raise ValueError("Expanded image prompt plus completion budget exceeds context; truncation is forbidden")
    ModelInput, EncodedTextChunk, ImageChunk = _chunk_types(chunk_types)
    prompt = ModelInput(chunks=[
        EncodedTextChunk(tokens=prefix_ids),
        ImageChunk(data=prepared, format=image_format, expected_tokens=geometry.expected_tokens),
        EncodedTextChunk(tokens=suffix_ids),
    ])
    if prompt.length != len(expected):
        raise ValueError("ModelInput chunk lengths differ from expected expanded token length")
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    audit.update({
        "prompt_tokens": len(expected), "enable_thinking": True,
        "image_token_id": image_id[0],
        "image_target_span": [len(prefix_ids), len(prefix_ids) + geometry.expected_tokens],
        "processor_fingerprint": hashlib.sha256(config_json.encode()).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(expanded.encode()).hexdigest(),
    })
    return prompt, expected, audit


prepare = build_prompt


def validate_sampled_prompt(response: Any, expected_prompt_ids: list[int]) -> None:
    """Require the sampler's image expansion to match every audited token."""
    resolved = (response.get("prompt_token_ids") if isinstance(response, dict)
                else getattr(response, "prompt_token_ids", None))
    if resolved is None:
        raise ValueError("Sampler omitted resolved image prompt IDs; cannot verify image alignment")
    if not isinstance(resolved, list) or any(type(token) is not int for token in resolved):
        raise ValueError("Sampler returned invalid resolved prompt IDs")
    if len(resolved) != len(expected_prompt_ids):
        raise ValueError(f"Sampler image expansion length mismatch: {len(resolved)} versus {len(expected_prompt_ids)}")
    mismatch = next((index for index, (actual, expected) in
                     enumerate(zip(resolved, expected_prompt_ids)) if actual != expected), None)
    if mismatch is not None:
        raise ValueError(f"Sampler image prompt differs from processor tokenization at position {mismatch}")


validate_resolved_prompt = validate_sampled_prompt


def model_input_with_completion(prompt: Any, completion_tokens: list[int], *,
                                next_token_inputs: bool = False, chunk_types=None) -> Any:
    """Append generated IDs while retaining image chunks for any loss callback.

    Native policy losses take the full completion. A custom causal-label loss
    can request removal of the last input token without flattening media.
    """
    if any(type(token) is not int or token < 0 for token in completion_tokens):
        raise ValueError("Completion contains invalid token IDs")
    if next_token_inputs and not completion_tokens:
        raise ValueError("A causal input/target split requires a completion token")
    tokens = completion_tokens[:-1] if next_token_inputs else list(completion_tokens)
    ModelInput, EncodedTextChunk, _ = _chunk_types(chunk_types)
    chunks = list(prompt.chunks)
    if tokens:
        chunks.append(EncodedTextChunk(tokens=tokens))
    return ModelInput(chunks=chunks)
