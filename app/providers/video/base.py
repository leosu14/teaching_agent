"""Video composer and prober contracts.

A VideoComposer turns a validated VideoPlan into a video file in a workspace directory it is given. It only reads
the image and audio bytes the plan already references, through the media reader it was given, and checks each one
against the checksum the plan recorded; it never generates an image, synthesizes speech or calls an AI video
service, and it has no LLM cost. Which command-line tool (if any) does the work is the composer's business: callers
see only `compose(plan, workspace) -> ComposedVideo`.

A VideoProber measures a file with a real container parser, so validation never trusts a file name or extension.

A VideoNormalizer converts a generated clip (whatever a video generation provider returned) into the platform's
format (codec, container, resolution, frame rate, audio or none), so the composer only ever sees platform-standard
clips and never anything vendor-specific.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from app.schemas.video import (
    AudioWindow,
    ClipNormalization,
    ComposedVideo,
    FrameSample,
    NormalizedClip,
    VideoPlan,
    VideoProbe,
)

MediaReader = Callable[[str], bytes]  # object uri -> bytes


class VideoCompositionError(Exception):
    pass


class VideoProbeError(Exception):
    pass


class VideoComposer(ABC):
    name: str  # name/version: part of the composition key, so a new composer version composes again

    @abstractmethod
    def compose(self, plan: VideoPlan, workspace: Path) -> ComposedVideo: ...


class VideoProber(ABC):
    name: str

    @abstractmethod
    def probe(self, path: Path) -> VideoProbe:
        """Container, streams, duration. Raises VideoProbeError for anything that is not readable video."""

    @abstractmethod
    def audio_levels(self, path: Path, windows: list[AudioWindow]) -> dict[str, float]:
        """Peak level of the audio in each window, in dBFS (-inf for digital silence)."""

    @abstractmethod
    def frame_stats(self, path: Path, samples: list[FrameSample]) -> dict[str, FrameStats]:
        """A small fingerprint of the frame shown at each sample time."""

    def decode_errors(self, path: Path) -> list[str]:
        """Errors met when decoding every frame of the file (empty: it decodes cleanly). Probers that cannot decode
        report nothing; the FFmpeg prober decodes the whole file."""
        return []


class VideoNormalizationError(Exception):
    pass


class VideoNormalizer(ABC):
    name: str  # name/version: part of a generated asset's identity

    @abstractmethod
    def normalize(self, source: Path, probe: VideoProbe, target: ClipNormalization,
                  workspace: Path) -> NormalizedClip:
        """Write the platform-format version of `source` (already probed) into `workspace`."""


class FrameStats:
    """Luma statistics of one frame, downscaled: enough to tell an empty frame and a slide change, not pixels."""

    __slots__ = ("mean", "stddev", "thumbnail")

    def __init__(self, mean: float, stddev: float, thumbnail: bytes) -> None:
        self.mean, self.stddev, self.thumbnail = mean, stddev, thumbnail

    def differs_from(self, other: FrameStats, threshold: float = 2.0) -> bool:
        """Mean absolute difference of the thumbnails, in luma levels."""
        if len(self.thumbnail) != len(other.thumbnail) or not self.thumbnail:
            return True
        diff = sum(abs(a - b) for a, b in zip(self.thumbnail, other.thumbnail)) / len(self.thumbnail)
        return diff > threshold


def read_media(media: MediaReader, uri: str, checksum: str, what: str) -> bytes:
    """Stored bytes a plan references, verified against the checksum the asset recorded."""
    try:
        data = media(uri)
    except (OSError, ValueError) as exc:
        raise VideoCompositionError(f"{what} cannot be read: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != checksum:
        raise VideoCompositionError(f"{what} does not match its recorded checksum")
    return data
