"""Image bytes helpers: a deterministic PNG encoder (for mock providers and fixtures) and a header probe that
reads an image's real format and dimensions (for validation). Standard library only."""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ImageProbe:
    format: str  # short name: png, jpeg, gif, webp, svg
    media_type: str
    width: int
    height: int


class ImageProbeError(ValueError):
    pass


def encode_png(width: int, height: int, colors: list[tuple[int, int, int]]) -> bytes:
    """An RGB PNG of horizontal bands in `colors`. Same arguments, same bytes."""
    if width < 1 or height < 1 or not colors:
        raise ValueError("width, height and at least one colour are required")
    band = max(1, height // len(colors))
    rows = []
    for y in range(height):
        r, g, b = colors[min(y // band, len(colors) - 1)]
        rows.append(b"\x00" + bytes((r, g, b)) * width)
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def colors_from(digest: bytes, n: int = 3) -> list[tuple[int, int, int]]:
    """`n` colours taken from a hash digest (at least 3*n bytes)."""
    return [(digest[3 * i], digest[3 * i + 1], digest[3 * i + 2]) for i in range(n)]


_SVG_DIM = re.compile(r'\b(width|height)\s*=\s*"([\d.]+)(?:px)?"')
_SVG_VIEWBOX = re.compile(r'\bviewBox\s*=\s*"\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"')


def probe_image(data: bytes) -> ImageProbe:
    """Format and pixel dimensions read from the bytes themselves, never from metadata."""
    if data.startswith(PNG_SIGNATURE) and len(data) >= 24 and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return ImageProbe("png", "image/png", width, height)
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return ImageProbe("gif", "image/gif", width, height)
    if data[:2] == b"\xff\xd8":
        return _probe_jpeg(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _probe_webp(data)
    head = data[:4096].decode("utf-8", errors="ignore")
    if "<svg" in head:
        dims = dict(_SVG_DIM.findall(head[head.index("<svg"):].split(">", 1)[0]))
        if "width" in dims and "height" in dims:
            return ImageProbe("svg", "image/svg+xml", round(float(dims["width"])), round(float(dims["height"])))
        box = _SVG_VIEWBOX.search(head)
        if box:
            return ImageProbe("svg", "image/svg+xml", round(float(box.group(1))), round(float(box.group(2))))
        raise ImageProbeError("SVG without width/height or viewBox")
    raise ImageProbeError("unrecognised image format")


def _probe_jpeg(data: bytes) -> ImageProbe:
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            raise ImageProbeError("corrupt JPEG marker stream")
        marker = data[i + 1]
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return ImageProbe("jpeg", "image/jpeg", width, height)
        i += 2 + length
    raise ImageProbeError("JPEG without a frame header")


def _probe_webp(data: bytes) -> ImageProbe:
    kind = data[12:16]
    if kind == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
    elif kind == b"VP8 " and len(data) >= 30:
        width, height = (v & 0x3FFF for v in struct.unpack("<HH", data[26:30]))
    elif kind == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        width, height = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    else:
        raise ImageProbeError("unsupported WebP variant")
    return ImageProbe("webp", "image/webp", width, height)
