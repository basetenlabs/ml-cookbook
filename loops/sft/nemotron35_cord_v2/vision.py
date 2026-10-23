"""NVIDIA image geometry and lossless PNG transport for CORD-v2 SFT.

The grid follows the audited local processor configuration, not an asserted
managed processor identity. Float bicubic output needs explicit uint8 quantization.
Input/target alignment is audited in data.py; server compatibility needs smoke.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROCESSOR_CONFIG = (
    Path(__file__).resolve().parent / "assets" / "preprocessor_config.json"
)
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
    required = {
        "patch_size",
        "downsample_ratio",
        "min_num_patches",
        "max_num_patches",
        "max_model_len",
    }
    if not required.issubset(config):
        raise ValueError("Nemotron image processor configuration is incomplete")
    for key in ("patch_size", "min_num_patches", "max_num_patches", "max_model_len"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid Nemotron processor setting: {key}")
    ratio = config["downsample_ratio"]
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid Nemotron downsample ratio")
    factor = round(1.0 / ratio)
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
        return (
            self.width_patches * self.patch_size,
            self.height_patches * self.patch_size,
        )

    @property
    def expected_tokens(self) -> int:
        return self.width_patches * self.height_patches // self.downsample_factor**2

    @property
    def needs_resize(self) -> bool:
        return self.prepared_size != (self.original_width, self.original_height)


def derive_image_geometry(
    width: int, height: int, config: dict | None = None
) -> ImageGeometry:
    """Reproduce the pinned processor's single-image dynamic patch grid.

    Python's round(x + .5), including ties-to-even, is intentional. It must
    not be replaced with ceil or with a Qwen image-grid rule.
    """
    config = load_processor_config() if config is None else config
    _validate_processor_config(config)
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    patch = config["patch_size"]
    divisor = round(1.0 / config["downsample_ratio"])
    minimum = config["min_num_patches"]
    budget = max((config["max_model_len"] - 4) * divisor**2, minimum)
    budget = max(min(budget, config["max_num_patches"]), minimum)
    closest_h = round(height / patch + 0.5)
    closest_w = round(width / patch + 0.5)
    if closest_h <= 0 or closest_w <= 0:
        raise ValueError("Image is too small for the reference patch rule")
    factor = min(math.sqrt(budget / (closest_h * closest_w)), 1.0)
    target_h = math.floor(factor * closest_h)
    target_w = math.floor(factor * closest_w)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Image aspect ratio produces an empty reference patch grid")
    if budget > minimum and target_h * target_w < minimum:
        up = math.sqrt(minimum / (target_h * target_w))
        target_h = math.ceil(up * target_h)
        target_w = math.ceil(up * target_w)
    rem_h = target_h % divisor
    if rem_h:
        inc_h = divisor - rem_h
        target_h = (
            target_h + inc_h
            if (target_h + inc_h) * target_w <= budget
            else max(divisor, target_h - rem_h)
        )
    rem_w = target_w % divisor
    if rem_w:
        inc_w = divisor - rem_w
        target_w = (
            target_w + inc_w
            if target_h * (target_w + inc_w) <= budget
            else max(divisor, target_w - rem_w)
        )
    if target_h * target_w > budget or min(target_h, target_w) < divisor:
        raise ValueError("Reference image grid violates the available patch budget")
    return ImageGeometry(width, height, target_w, target_h, patch, divisor, budget)


def _prepare_image(
    raw: bytes, declared_format: str, config: dict, allow_quantized_resize: bool
) -> tuple[bytes, str, ImageGeometry, dict]:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as opened:
        actual_format = {"PNG": "png", "JPEG": "jpeg"}.get(opened.format)
        if actual_format != declared_format:
            raise ValueError("Inline image MIME type does not match its decoded format")
        width, height = opened.size
        if width * height > 50_000_000:
            raise ValueError("Source image exceeds the safe local decode limit")
        geometry = derive_image_geometry(width, height, config)
        prepared_width, prepared_height = geometry.prepared_size
        if prepared_width * prepared_height * 3 > 18 * 1024 * 1024:
            raise ValueError("Prepared RGB image exceeds the Loops image limit")
        original_mode = opened.mode
        image = opened.convert("RGB")
        image.load()
    if geometry.needs_resize and not allow_quantized_resize:
        raise ValueError(
            f"Image {width}x{height} requires reference resize to "
            f"{geometry.prepared_size}; Loops PNG quantization differs from float "
            "reference pixels. Explicitly enable allow_quantized_resize."
        )
    if geometry.needs_resize:
        import numpy as np
        import torch

        array = np.asarray(image, dtype=np.uint8).copy()
        pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        target_w, target_h = geometry.prepared_size
        pixels = torch.nn.functional.interpolate(
            pixels,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
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
    if (
        stable.prepared_size != geometry.prepared_size
        or stable.expected_tokens != geometry.expected_tokens
    ):
        raise ValueError("Prepared image would change grid when processed again")
    audit = {
        "original_size": [width, height],
        "prepared_size": list(geometry.prepared_size),
        "original_image_sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_image_sha256": hashlib.sha256(prepared).hexdigest(),
        "image_format": image_format,
        "expected_image_tokens": geometry.expected_tokens,
        "patch_grid": [geometry.width_patches, geometry.height_patches],
        "exact_image_pixels": not geometry.needs_resize,
        "pixel_fidelity": (
            "quantized_after_reference_float_resize"
            if geometry.needs_resize
            else "reference_rgb_pixels_preserved"
        ),
        "resize_kernel": "torch_bicubic_align_corners_false_antialias_true",
        "managed_processor_parity_verified": False,
    }
    return prepared, image_format, geometry, audit
