"""Offline PNG compression and frozen first-row regression tests."""

import base64
import hashlib
from pathlib import Path
import struct
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canonical_png import canonical_png  # noqa: E402


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def make_png(raw=b"\x00\x11\x22\x33", width=1, height=1, extra=b"", fragmented=False):
    header = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    encoded = zlib.compress(raw, 1)
    blocks = [encoded[:3], encoded[3:]] if fragmented else [encoded]
    return (b"\x89PNG\r\n\x1a\n" + header + extra
            + b"".join(chunk(b"IDAT", block) for block in blocks) + chunk(b"IEND", b""))


def png_image(row):
    url = row["responses_create_params"]["input"][1]["content"][0]["image_url"]
    return base64.b64decode(url.split(",", 1)[1])


class CanonicalPngTests(unittest.TestCase):
    def test_preserves_header_metadata_end_and_filtered_scanlines(self):
        raw = b"\x00\x11\x22\x33"
        metadata = chunk(b"tEXt", b"comment\x00unchanged")
        png = make_png(raw, extra=metadata, fragmented=True)
        actual = canonical_png(png)
        encoder = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_FILTERED)
        expected_idat = encoder.compress(raw) + encoder.flush()
        self.assertEqual(actual, png[:33] + metadata + chunk(b"IDAT", expected_idat) + png[-12:])
        self.assertIs(canonical_png(actual), actual)

    def test_invalid_bytes_fail_closed(self):
        good = make_png()
        cases = [b"not PNG", good[:-1], good + b"trailing", good[:20] + b"x" + good[21:],
                 make_png(b"wrong length"), make_png(width=1000000, height=1000000)]
        for value in cases:
            with self.subTest(length=len(value)), self.assertRaises(ValueError):
                canonical_png(value)
        with self.assertRaises(TypeError):
            canonical_png(bytearray(good))

    def test_disjoint_idat_is_rejected(self):
        png = make_png(fragmented=True)
        first_stop = 33 + 12 + 3
        png = png[:first_stop] + chunk(b"tEXt", b"x\x00y") + png[first_stop:]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            canonical_png(png)

    def test_frozen_first_train_and_validation_images_are_byte_exact(self):
        from generate_data import make_example
        expected = (
            ("train", 0, "9267044c1fbb35fe95a058e04d708fa7696feaf8155fd1910978650651622574"),
            ("validation", 1000000, "769d02eb3808ca4263e6673dd9ac1558e790d43c08110da911ca387569629117"),
        )
        for split, seed, checksum in expected:
            with self.subTest(split=split):
                generated = png_image(make_example(seed))
                self.assertEqual(hashlib.sha256(canonical_png(generated)).hexdigest(), checksum)
                self.assertIs(canonical_png(generated), generated)


if __name__ == "__main__":
    unittest.main()
