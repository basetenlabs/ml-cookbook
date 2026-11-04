"""Canonicalize the frozen RGB PNGs without changing pixels or PNG filters.

Pillow 12.3 optimize=True uses level=9, wbits=15, memLevel=9 and Z_FILTERED
for these RGB images. Pillow's macOS wheel uses zlib-ng, whose compressed
bytes differ. Recompress its filtered scanlines using Python's regular zlib;
the generator's frozen whole-dataset SHA256 checks remain the final guard.
This module never downloads, builds, reads files, or changes global settings.
"""

from __future__ import annotations

import struct
import zlib


_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IDAT_SIZE = 65536  # Pillow's encoder buffer for the frozen <=1200px images.
_MAX_FILTERED_BYTES = 32 * 1024 * 1024


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def canonical_png(png: bytes) -> bytes:
    """Return canonical RGB8/noninterlaced PNG bytes, preserving other chunks.

    PNG checksums, framing, dimensions and decompressed length are validated.
    Only the IDAT deflate stream and its chunk checksums are replaced. Already
    canonical bytes are returned unchanged, including the original bytes object.
    """
    if not isinstance(png, bytes):
        raise TypeError("canonical_png requires bytes")
    if not png.startswith(_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    chunks = []
    position = len(_SIGNATURE)
    seen_idat = ended_idat = False
    while position < len(png):
        if len(png) - position < 12:
            raise ValueError("Truncated PNG chunk")
        size = struct.unpack_from(">I", png, position)[0]
        stop = position + size + 12
        if stop > len(png):
            raise ValueError("Truncated PNG chunk data")
        kind, data = png[position + 4:position + 8], png[position + 8:stop - 4]
        expected_crc = struct.unpack_from(">I", png, stop - 4)[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            raise ValueError("Invalid PNG chunk checksum")
        if kind == b"IDAT":
            if ended_idat:
                raise ValueError("PNG IDAT chunks must be contiguous")
            seen_idat = True
        elif seen_idat:
            ended_idat = True
        chunks.append((kind, data, png[position:stop]))
        position = stop
        if kind == b"IEND":
            if data or position != len(png):
                raise ValueError("Invalid PNG end or trailing bytes")
            break
    if (not chunks or chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13
            or chunks[-1][0] != b"IEND" or not seen_idat
            or sum(kind == b"IHDR" for kind, _, _ in chunks) != 1):
        raise ValueError("Invalid PNG chunk structure")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1])
    if not width or not height or (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("Canonical encoder requires noninterlaced RGB8 PNG")
    expected_size = height * (width * 3 + 1)
    if expected_size > _MAX_FILTERED_BYTES:
        raise ValueError("PNG exceeds canonical encoder size limit")
    compressed = b"".join(data for kind, data, _ in chunks if kind == b"IDAT")
    try:
        decoder = zlib.decompressobj()
        filtered = decoder.decompress(compressed, expected_size + 1)
    except zlib.error:
        raise ValueError("Invalid PNG compressed stream") from None
    if (len(filtered) != expected_size or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("PNG compressed stream length or framing mismatch")
    encoder = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_FILTERED)
    compressed = encoder.compress(filtered) + encoder.flush()
    replacement = b"".join(_chunk(b"IDAT", compressed[offset:offset + _IDAT_SIZE])
                           for offset in range(0, len(compressed), _IDAT_SIZE))
    output = [_SIGNATURE]
    replaced = False
    for kind, _, original in chunks:
        if kind != b"IDAT":
            output.append(original)
        elif not replaced:
            output.append(replacement)
            replaced = True
    result = b"".join(output)
    return png if result == png else result
