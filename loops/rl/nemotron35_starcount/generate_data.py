#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the frozen star-count train/validation JSONL files offline.

The seeded five-point-star renderer uses two non-overlapping seed ranges,
canonical PNG compression, pinned hashes, caching, and no-clobber publication.
No model, credentials, network access, or managed compute is needed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any
import zlib

import PIL
from PIL import Image, ImageDraw, features

from canonical_png import canonical_png
from common import FROZEN_SPLITS


ROOT = Path(__file__).resolve().parent
DATASET_VERSION = "starcount-v1"
GENERATOR = "seeded-five-point-stars-rgb8"
SPLITS = {
    "train": {"rows": 1024, "seed_offset": 0, "sha256": FROZEN_SPLITS["train"]["sha256"]},
    "validation": {"rows": 256, "seed_offset": 1_000_000,
                   "sha256": FROZEN_SPLITS["validation"]["sha256"]},
}
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (133, 153, 0),
    "yellow": (181, 137, 0),
    "purple": (108, 113, 196),
    "orange": (203, 75, 22),
    "cyan": (42, 161, 152),
    "pink": (211, 54, 130),
}
AGENT_REF = {"type": "responses_api_agents", "name": "circle_count_simple_agent"}
SYSTEM_PROMPT = (
    "You are a visual assistant. Count the number of stars of the specified "
    "color in the image. Output your final answer in \\boxed{} format, e.g. "
    "\\boxed{3}."
)


def _place_stars(count: int, canvas_size: int, radius: int,
                 rng: random.Random) -> list[tuple[int, int]]:
    margin = radius + 10
    positions: list[tuple[int, int]] = []
    minimum_distance = 2 * radius + 12
    for _ in range(count):
        for _ in range(2_000):
            point = (rng.randint(margin, canvas_size - margin),
                     rng.randint(margin, canvas_size - margin))
            if all(math.dist(point, other) > minimum_distance for other in positions):
                positions.append(point)
                break
        else:
            raise RuntimeError(f"could not place {count} stars on a {canvas_size}px "
                               f"canvas with radius {radius}")
    return positions


def _star_vertices(x: int, y: int, radius: int) -> list[tuple[float, float]]:
    vertices = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        point_radius = radius if index % 2 == 0 else radius * 0.42
        vertices.append((x + point_radius * math.cos(angle),
                         y + point_radius * math.sin(angle)))
    return vertices


def _render(stars: list[dict[str, Any]], canvas_size: int) -> str:
    image = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for star in stars:
        draw.polygon(_star_vertices(star["x"], star["y"], star["radius"]),
                     fill=COLORS[star["color"]])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(canonical_png(buffer.getvalue())).decode()


def make_example(
    seed: int,
    canvas_size_range: tuple[int, int] = (800, 1200),
    star_radius_range: tuple[int, int] = (24, 48),
    num_stars_range: tuple[int, int] = (1, 30),
    num_colors_range: tuple[int, int] = (2, 4),
) -> dict[str, Any]:
    """Render one deterministic example using the fixed RNG draw order."""
    rng = random.Random(seed)
    canvas_size = rng.randint(*canvas_size_range)
    radius = rng.randint(*star_radius_range)
    num_stars = rng.randint(*num_stars_range)
    num_colors = min(num_stars, rng.randint(*num_colors_range))
    palette = rng.sample(list(COLORS), num_colors)
    color_names = palette + [rng.choice(palette) for _ in range(num_stars - num_colors)]
    rng.shuffle(color_names)
    target_color = rng.choice(palette)
    positions = _place_stars(num_stars, canvas_size, radius, rng)
    stars = [{"x": x, "y": y, "radius": radius, "color": color}
             for (x, y), color in zip(positions, color_names, strict=True)]
    # The frozen wire-format key is "circles" even though the polygons are stars.
    return {
        "responses_create_params": {"input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "input_image", "image_url": _render(stars, canvas_size), "detail": "auto"},
                {"type": "input_text", "text": f"How many {target_color} stars are in the image?"},
            ]},
        ]},
        "circles": stars,
        "target_color": target_color,
        "agent_ref": dict(AGENT_REF),
    }


def _identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            rows += 1
    return digest.hexdigest(), rows


def _check_existing(path: Path, spec: dict[str, Any]) -> bool:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink dataset: {path}")
    if not path.exists():
        return False
    if not path.is_file() or _identity(path) != (spec["sha256"], spec["rows"]):
        raise ValueError(f"Existing dataset differs from the pinned dataset: {path}. "
                         "It was not overwritten; use a new --output-dir.")
    return True


def generate_dataset(output_dir: Path, *, rebuild: bool = False) -> dict[str, Any]:
    """Build or reuse the two frozen splits; never replace an existing file.

    All generated splits must match their pinned JSONL hashes (including PNG
    bytes) before any new split is published. --rebuild checks regeneration
    even when correct cached files already exist, leaving those files intact.
    Publication is atomic per file, not a two-file transaction: an I/O error
    may leave one verified split cached, which the next invocation can reuse.
    """
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError(f"Refusing symlink output directory: {output_dir}")
    existing = {name: _check_existing(output_dir / f"{name}.jsonl", spec)
                for name, spec in SPLITS.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    to_build = [name for name in SPLITS if rebuild or not existing[name]]
    if to_build:
        # Same filesystem permits atomic, no-clobber hard links at publication.
        with tempfile.TemporaryDirectory(prefix=".starcount-build-", dir=output_dir) as scratch:
            built = {}
            for name in to_build:
                spec = SPLITS[name]
                temporary = Path(scratch) / f"{name}.jsonl"
                print(f"Generating {name}: {spec['rows']} rows, seed offset "
                      f"{spec['seed_offset']}", file=sys.stderr, flush=True)
                with temporary.open("w", encoding="utf-8", newline="\n") as output:
                    for index in range(spec["rows"]):
                        output.write(json.dumps(make_example(spec["seed_offset"] + index)) + "\n")
                if _identity(temporary) != (spec["sha256"], spec["rows"]):
                    raise ValueError(f"Regenerated {name} does not match the frozen dataset. "
                                     f"No generated split was installed. Use the locked environment "
                                     f"(Pillow 12.3.0); current zlib is {zlib.ZLIB_RUNTIME_VERSION}. "
                                     "Do not update the expected hashes to bypass this check.")
                built[name] = temporary
            # Detect changes during generation before publishing anything.
            for name, spec in SPLITS.items():
                existing[name] = _check_existing(output_dir / f"{name}.jsonl", spec)
            for name, temporary in built.items():
                if not existing[name]:
                    os.link(temporary, output_dir / f"{name}.jsonl")
    return {
        "splits": {name: {**spec, "path": str(output_dir / f"{name}.jsonl"),
                          "status": "regenerated_and_verified" if rebuild and existing[name]
                          else "reused" if existing[name] else "generated"}
                   for name, spec in SPLITS.items()},
        "dataset_version": DATASET_VERSION,
        "generator": GENERATOR,
        "pillow_version": PIL.__version__,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        "pillow_zlib_version": features.version("zlib"),
        "compute_provisioned": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--rebuild", action="store_true",
                        help="Regenerate and verify both splits, without replacing correct cached files.")
    args = parser.parse_args(argv)
    try:
        report = generate_dataset(args.output_dir, rebuild=args.rebuild)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Data generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
