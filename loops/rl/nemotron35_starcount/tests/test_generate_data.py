"""Small offline generator and no-clobber publication checks; no full rebuild."""

import base64
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_data  # noqa: E402


def encoded_rows(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def tiny_example(seed):
    # Non-ASCII content also checks the fixed json.dumps defaults.
    return {"seed": seed, "description": "synthetic \N{LATIN SMALL LETTER E WITH ACUTE}"}


def tiny_splits():
    result = {}
    for name, count, offset in (("train", 3, 0), ("validation", 2, 1_000_000)):
        content = encoded_rows(tiny_example(offset + index) for index in range(count))
        result[name] = {"rows": count, "seed_offset": offset,
                        "sha256": hashlib.sha256(content).hexdigest()}
    return result


def image_from_row(row):
    messages = row["responses_create_params"]["input"]
    images = [part for message in messages if isinstance(message["content"], list)
              for part in message["content"] if part["type"] == "input_image"]
    if len(images) != 1:
        raise AssertionError("Expected exactly one embedded PNG.")
    prefix, payload = images[0]["image_url"].split(",", 1)
    if prefix != "data:image/png;base64":
        raise AssertionError("Expected the frozen inline PNG wire format.")
    image = Image.open(BytesIO(base64.b64decode(payload, validate=True)))
    image.load()
    return image


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "common.api_request", "common.load_env"):
            guard = patch(name, side_effect=AssertionError("Generator must remain offline and credential-free"))
            guard.start()
            self.addCleanup(guard.stop)


class ExampleGenerationTests(OfflineTestCase):
    def test_first_and_last_examples_match_both_packaged_splits_exactly(self):
        # Hashes of the original frozen rows, including their embedded PNGs.
        # Unit tests must also work before downloading assets or generating data.
        expected = {
            ("train", 0): "6a001403d526b4d4fc7ae54d906f3f7b6f7df3f64597b062a2ea946f75185207",
            ("train", 1023): "ddec1beae6b8bf74a1a2f45be0ad6bfdb7a4cb20a0f7330c4e805a11d1341718",
            ("validation", 0): "9c2c39431808c612523881cfd13307f4b6a96f0583efa02ad3b6cba3c41b0597",
            ("validation", 255): "dfdef74979cbdb9db0c73f174082810a4a3aaa752a0108b4256de73ba1bf6ede",
        }
        for (name, index), checksum in expected.items():
            with self.subTest(split=name, index=index):
                seed = generate_data.SPLITS[name]["seed_offset"] + index
                actual = generate_data.make_example(seed)
                actual_hash = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
                self.assertEqual(actual_hash, checksum)

    def test_seed_is_deterministic_independent_of_global_rng(self):
        before = random.getstate()
        first = generate_data.make_example(31)
        self.assertEqual(random.getstate(), before)
        try:
            random.seed(123456)
            random.random()
            self.assertEqual(generate_data.make_example(31), first)
        finally:
            random.setstate(before)
        self.assertNotEqual(generate_data.make_example(32), first)

    def test_geometry_pixels_metadata_and_positive_ground_truth(self):
        for seed in (0, 23, 1023, 1_000_000, 1_000_255):
            with self.subTest(seed=seed):
                row = generate_data.make_example(seed)
                image = image_from_row(row)
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.width, image.height)
                self.assertTrue(800 <= image.width <= 1200)
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
                stars = row["circles"]
                self.assertTrue(1 <= len(stars) <= 30)
                self.assertEqual(row["agent_ref"],
                    {"type": "responses_api_agents", "name": "circle_count_simple_agent"})
                self.assertGreater(sum(star["color"] == row["target_color"] for star in stars), 0)
                messages = row["responses_create_params"]["input"]
                self.assertEqual([message["role"] for message in messages], ["system", "user"])
                question = next(part["text"] for part in messages[1]["content"]
                                if part["type"] == "input_text")
                self.assertEqual(question, f"How many {row['target_color']} stars are in the image?")
                for index, star in enumerate(stars):
                    x, y, radius = star["x"], star["y"], star["radius"]
                    self.assertTrue(24 <= radius <= 48)
                    self.assertTrue(radius + 10 <= x <= image.width - radius - 10)
                    self.assertTrue(radius + 10 <= y <= image.height - radius - 10)
                    self.assertEqual(image.getpixel((x, y)), generate_data.COLORS[star["color"]])
                    # Five-point stars do not fill the bottom of their bounding
                    # circle. This also catches accidentally drawing circles.
                    self.assertEqual(image.getpixel((x, y + radius - 2)), (255, 255, 255))
                    for other in stars[:index]:
                        self.assertGreater(math.dist((x, y), (other["x"], other["y"])),
                                           radius + other["radius"] + 12)

    def test_custom_generation_ranges_are_honored(self):
        row = generate_data.make_example(9, canvas_size_range=(200, 200),
            star_radius_range=(10, 10), num_stars_range=(3, 3), num_colors_range=(2, 2))
        self.assertEqual(image_from_row(row).size, (200, 200))
        self.assertEqual(len(row["circles"]), 3)
        self.assertEqual({star["radius"] for star in row["circles"]}, {10})
        self.assertEqual(len({star["color"] for star in row["circles"]}), 2)

    def test_frozen_split_identity_and_seed_ranges_do_not_overlap(self):
        expected = {
            "train": (1024, 0, "6b63bf98dd2107a35b6ed442dbb5a8003ba15c8053b66dc37f9fbe617d24b307"),
            "validation": (256, 1_000_000, "76a8a383ab459826117bbf11699c5908b10d539d9fd21b437d0c1e9149f096b8"),
        }
        ranges = []
        self.assertEqual(set(generate_data.SPLITS), set(expected))
        for name, identity in expected.items():
            spec = generate_data.SPLITS[name]
            self.assertEqual((spec["rows"], spec["seed_offset"], spec["sha256"]), identity)
            ranges.append(set(range(spec["seed_offset"], spec["seed_offset"] + spec["rows"])))
        self.assertFalse(ranges[0] & ranges[1])


class DatasetPublicationTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        temporary = TemporaryDirectory(prefix="starcount-generator-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "data"
        self.specs = tiny_splits()
        split_patch = patch.object(generate_data, "SPLITS", self.specs)
        split_patch.start()
        self.addCleanup(split_patch.stop)
        row_patch = patch.object(generate_data, "make_example", side_effect=tiny_example)
        self.factory = row_patch.start()
        self.addCleanup(row_patch.stop)

    def populate(self, *names):
        self.output.mkdir(exist_ok=True)
        for name in names:
            spec = self.specs[name]
            content = encoded_rows(tiny_example(spec["seed_offset"] + index)
                                   for index in range(spec["rows"]))
            (self.output / f"{name}.jsonl").write_bytes(content)

    def snapshot(self):
        return {name: ((self.output / f"{name}.jsonl").read_bytes(),
                       (self.output / f"{name}.jsonl").stat().st_mtime_ns)
                for name in self.specs if (self.output / f"{name}.jsonl").is_file()}

    def run_generator(self, **kwargs):
        with redirect_stderr(StringIO()):
            return generate_data.generate_dataset(self.output, **kwargs)

    def assert_report(self, report):
        self.assertIs(report["compute_provisioned"], False)
        self.assertEqual(report["dataset_version"], "starcount-v1")
        self.assertEqual(report["generator"], "seeded-five-point-stars-rgb8")
        self.assertNotIn("generator_source_sha256", report)
        self.assertNotIn("generator_source_revision", report)
        self.assertTrue(report["pillow_zlib_version"])
        self.assertEqual(set(report["splits"]), set(self.specs))
        for name, spec in self.specs.items():
            self.assertEqual({key: report["splits"][name][key] for key in spec}, spec)
            self.assertTrue(report["splits"][name]["status"])

    def test_missing_splits_use_distinct_deterministic_seeds_and_exact_jsonl(self):
        self.assert_report(self.run_generator())
        self.assertEqual(self.factory.call_args_list,
                         [call(0), call(1), call(2), call(1_000_000), call(1_000_001)])
        seen = []
        for name, spec in self.specs.items():
            content = (self.output / f"{name}.jsonl").read_bytes()
            expected = encoded_rows(tiny_example(spec["seed_offset"] + index)
                                    for index in range(spec["rows"]))
            self.assertEqual(content, expected)
            self.assertEqual(hashlib.sha256(content).hexdigest(), spec["sha256"])
            seen.append({json.loads(line)["seed"] for line in content.splitlines()})
        self.assertFalse(seen[0] & seen[1])

    def test_correct_cached_splits_are_reused_without_rendering_or_rewriting(self):
        self.populate("train", "validation")
        before = self.snapshot()
        self.assert_report(self.run_generator())
        self.factory.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_only_missing_split_is_generated(self):
        self.populate("train")
        before = self.snapshot()["train"]
        self.assert_report(self.run_generator())
        self.assertEqual(self.factory.call_args_list, [call(1_000_000), call(1_000_001)])
        self.assertEqual(self.snapshot()["train"], before)

    def test_rebuild_verifies_all_rows_without_replacing_correct_files(self):
        self.populate("train", "validation")
        before = self.snapshot()
        self.assert_report(self.run_generator(rebuild=True))
        self.assertEqual(self.factory.call_count, 5)
        self.assertEqual(self.snapshot(), before)

    def test_mismatching_existing_file_is_never_overwritten_with_or_without_rebuild(self):
        self.output.mkdir()
        path = self.output / "validation.jsonl"
        content = b"existing unrelated data\n"
        path.write_bytes(content)
        for rebuild in (False, True):
            with self.subTest(rebuild=rebuild), self.assertRaises(ValueError):
                self.run_generator(rebuild=rebuild)
            self.assertEqual(path.read_bytes(), content)
            self.assertFalse((self.output / "train.jsonl").exists())

    def test_wrong_generated_hash_publishes_neither_split(self):
        for name in self.specs:
            with self.subTest(bad_split=name):
                original = self.specs[name]["sha256"]
                self.specs[name]["sha256"] = "0" * 64
                try:
                    with self.assertRaises(ValueError):
                        self.run_generator()
                    self.assertFalse((self.output / "train.jsonl").exists())
                    self.assertFalse((self.output / "validation.jsonl").exists())
                finally:
                    self.specs[name]["sha256"] = original

    def test_failed_rebuild_keeps_all_correct_cached_bytes_and_timestamps(self):
        self.populate("train", "validation")
        before = self.snapshot()
        self.factory.side_effect = lambda seed: {"wrong_seed": seed}
        with self.assertRaises(ValueError):
            self.run_generator(rebuild=True)
        self.assertEqual(self.snapshot(), before)

    def test_changed_existing_file_during_generation_prevents_any_new_publication(self):
        self.output.mkdir()
        raced = self.output / "validation.jsonl"

        def concurrent_writer(seed):
            if not raced.exists():
                raced.write_bytes(b"concurrent writer owns this file\n")
            return tiny_example(seed)

        self.factory.side_effect = concurrent_writer
        with self.assertRaises(ValueError):
            self.run_generator()
        self.assertEqual(raced.read_bytes(), b"concurrent writer owns this file\n")
        self.assertFalse((self.output / "train.jsonl").exists())

    def test_existing_and_dangling_split_symlinks_are_refused_without_following(self):
        self.output.mkdir()
        target = self.root / "outside.jsonl"
        target.write_bytes(b"outside sentinel\n")
        link = self.output / "train.jsonl"
        for destination in (target, self.root / "absent.jsonl"):
            with self.subTest(destination=destination):
                link.symlink_to(destination)
                try:
                    with self.assertRaises(ValueError):
                        self.run_generator()
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(target.read_bytes(), b"outside sentinel\n")
                    self.assertFalse((self.output / "validation.jsonl").exists())
                finally:
                    link.unlink()

    def test_symlink_output_directory_is_refused_without_publishing_through_it(self):
        target = self.root / "outside-directory"
        target.mkdir()
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_generator()
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(list(target.iterdir()), [])

    def test_existing_directory_in_place_of_split_is_refused(self):
        directory = self.output / "train.jsonl"
        directory.mkdir(parents=True)
        sentinel = directory / "keep.txt"
        sentinel.write_text("do not remove")
        with self.assertRaises(ValueError):
            self.run_generator()
        self.assertEqual(sentinel.read_text(), "do not remove")
        self.assertFalse((self.output / "validation.jsonl").exists())

    def test_cli_generates_and_rebuilds_tiny_dataset_without_network(self):
        for rebuild in (False, True):
            args = ["--output-dir", str(self.output)] + (["--rebuild"] if rebuild else [])
            with self.subTest(rebuild=rebuild), redirect_stdout(StringIO()) as output, \
                    redirect_stderr(StringIO()):
                self.assertEqual(generate_data.main(args), 0)
                self.assert_report(json.loads(output.getvalue()))
        self.assertEqual(self.factory.call_count, 10)

    def test_cli_reports_generation_refusal_with_nonzero_status(self):
        self.output.mkdir()
        path = self.output / "train.jsonl"
        path.write_bytes(b"unrelated dataset\n")
        with redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()) as error:
            self.assertEqual(generate_data.main(["--output-dir", str(self.output)]), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Data generation failed", error.getvalue())
        self.assertEqual(path.read_bytes(), b"unrelated dataset\n")
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
