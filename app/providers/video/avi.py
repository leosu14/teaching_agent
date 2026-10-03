"""Motion-JPEG AVI: a tiny, deterministic, genuinely playable video container written and read without FFmpeg.

The mock video generation provider writes its fixture clips with `write_mjpeg_avi` (any player and FFmpeg can
decode them). `read_avi` parses the RIFF structure and every frame chunk from the bytes themselves, so the mock
prober can validate a clip (container, codec, size, frame rate, frame count, corruption) without FFmpeg, and a
truncated or corrupted file is detected rather than trusted.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

from PIL import Image

AVIF_HASINDEX = 0x10
AVIIF_KEYFRAME = 0x10


class AviError(ValueError):
    pass


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    pad = b"\x00" if len(data) % 2 else b""
    return fourcc + struct.pack("<I", len(data)) + data + pad


def _list(kind: bytes, payload: bytes) -> bytes:
    return b"LIST" + struct.pack("<I", 4 + len(payload)) + kind + payload


def jpeg_frame(image: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality, optimize=False, progressive=False, subsampling=2)
    return buf.getvalue()


def write_mjpeg_avi(frames: list[bytes], width: int, height: int, fps: int) -> bytes:
    """An AVI with one MJPEG video stream holding `frames` (JPEG bytes) at `fps`. Same frames, same bytes."""
    if not frames or width < 1 or height < 1 or fps < 1:
        raise AviError("an AVI needs frames, a size and a frame rate")
    max_frame = max(len(f) for f in frames)
    avih = struct.pack("<IIIIIIIIII4I", round(1_000_000 / fps), max_frame * fps, 0, AVIF_HASINDEX, len(frames), 0,
                       1, max_frame, width, height, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0, 0, 1, fps, 0, len(frames), max_frame,
                       0xFFFFFFFF, 0, 0, 0, width, height)
    strf = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG", width * height * 3, 0, 0, 0, 0)
    hdrl = _list(b"hdrl", _chunk(b"avih", avih) + _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)))
    movi_body, index, offset = b"", b"", 4  # idx1 offsets count from the 'movi' fourcc
    for frame in frames:
        chunk = _chunk(b"00dc", frame)
        index += b"00dc" + struct.pack("<III", AVIIF_KEYFRAME, offset, len(frame))
        movi_body += chunk
        offset += len(chunk)
    body = b"AVI " + hdrl + _list(b"movi", movi_body) + _chunk(b"idx1", index)
    return b"RIFF" + struct.pack("<I", len(body)) + body


@dataclass(frozen=True)
class AviInfo:
    codec: str  # e.g. "mjpeg"
    width: int
    height: int
    fps: float
    frames: int  # frames actually present (and intact) in the file
    declared_frames: int
    duration: float
    audio_streams: int


def is_avi(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI "


def read_avi(data: bytes) -> AviInfo:
    """Parse the headers and count intact frames. Raises AviError for anything malformed or truncated."""
    if not is_avi(data):
        raise AviError("not a RIFF AVI file")
    declared = struct.unpack("<I", data[4:8])[0] + 8
    if declared > len(data):
        raise AviError(f"truncated: the RIFF header declares {declared} bytes, the file has {len(data)}")
    width = height = frames_declared = 0
    rate = scale = 0
    codec = None
    audio = 0
    frames = 0
    pos = 12

    def walk(start: int, end: int) -> None:
        nonlocal width, height, frames_declared, rate, scale, codec, audio, frames
        p = start
        while p + 8 <= end:
            fourcc, size = data[p:p + 4], struct.unpack("<I", data[p + 4:p + 8])[0]
            body = p + 8
            if body + size > end:
                raise AviError(f"chunk {fourcc!r} at {p} runs past the end of its list")
            if fourcc == b"LIST":
                walk(body + 4, body + size)
            elif fourcc == b"avih" and size >= 40:
                frames_declared = struct.unpack("<I", data[body + 16:body + 20])[0]
                width, height = struct.unpack("<II", data[body + 32:body + 40])
            elif fourcc == b"strh" and size >= 36:
                kind, handler = data[body:body + 4], data[body + 4:body + 8]
                if kind == b"vids":
                    scale, rate = struct.unpack("<II", data[body + 20:body + 28])
                    codec = "mjpeg" if handler.upper() == b"MJPG" else handler.decode("latin-1").strip().lower()
                elif kind == b"auds":
                    audio += 1
            elif fourcc[2:] in (b"dc", b"db"):
                frame = data[body:body + size]
                if not (frame.startswith(b"\xff\xd8") and frame.rstrip(b"\x00").endswith(b"\xff\xd9")):
                    raise AviError(f"frame {frames + 1} is not an intact JPEG image")
                frames += 1
            p = body + size + (size % 2)

    walk(pos, declared)
    if codec is None or not rate or not scale:
        raise AviError("no video stream header")
    if not width or not height:
        raise AviError("no frame size")
    fps = rate / scale
    if frames == 0:
        raise AviError("no video frames")
    if frames_declared and frames != frames_declared:
        raise AviError(f"{frames} intact frames, but the header declares {frames_declared}")
    return AviInfo(codec=codec, width=width, height=height, fps=fps, frames=frames, declared_frames=frames_declared,
                   duration=frames / fps, audio_streams=audio)
