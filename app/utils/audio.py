"""Audio bytes helpers: a container probe that reads an audio file's real format, duration, sample rate and channels
(for validation), and a PCM WAV encoder (for mock providers and fixtures). Standard library only.

The probe recognises a container from its magic bytes and measures it with a reader for that container. Only WAV
has a reader so far; a recognised format without one (MP3, Ogg/Opus, FLAC) is reported as unmeasurable instead of
being trusted. Adding a format means adding a reader here, not changing callers.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioProbe:
    format: str  # short name: wav, mp3, ogg, flac
    media_type: str
    duration: float  # seconds
    sample_rate: int
    channels: int
    sample_width: int  # bytes per sample
    frames: int


class AudioProbeError(ValueError):
    pass


def sniff_format(content: bytes) -> str | None:
    if content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return "wav"
    if content[:3] == b"ID3" or (len(content) > 1 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0):
        return "mp3"
    if content[:4] == b"OggS":
        return "ogg"
    if content[:4] == b"fLaC":
        return "flac"
    return None


def _read_wav(content: bytes) -> AudioProbe:
    try:
        with wave.open(io.BytesIO(content), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            channels, width = wav.getnchannels(), wav.getsampwidth()
            data = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioProbeError(f"invalid WAV: {exc}") from exc
    if rate <= 0 or channels <= 0:
        raise AudioProbeError("invalid WAV header")
    if len(data) != frames * channels * width:
        raise AudioProbeError(f"WAV data is truncated: {len(data)} of {frames * channels * width} bytes")
    return AudioProbe(format="wav", media_type="audio/wav", duration=frames / rate, sample_rate=rate,
                      channels=channels, sample_width=width, frames=frames)


READERS: dict[str, Callable[[bytes], AudioProbe]] = {"wav": _read_wav}


def probe_audio(content: bytes) -> AudioProbe:
    if not content:
        raise AudioProbeError("no bytes")
    fmt = sniff_format(content)
    if fmt is None:
        raise AudioProbeError("not a recognised audio container")
    reader = READERS.get(fmt)
    if reader is None:
        raise AudioProbeError(f"{fmt} audio cannot be measured yet (no reader)")
    return reader(content)


def encode_wav(pcm: bytes, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """A PCM WAV file around raw little-endian samples. Same arguments, same bytes."""
    if len(pcm) % (channels * sample_width):
        raise ValueError("PCM data is not a whole number of frames")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()
