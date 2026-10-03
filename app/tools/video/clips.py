"""ClipValidator: checks a generated clip from its actual bytes, never from what the provider claimed.

The file must exist and be non-empty, match its checksum, be parsed by a real container parser (the configured
VideoProber: ffprobe, or the mock prober's RIFF/AVI parser in tests), decode without errors, and have a video stream
of the expected shape and length (and audio only where the plan wants it). A raw provider file is checked loosely
(its codec and size may be anything decodable and large enough); a normalised clip exactly (the platform's codec,
container, resolution and frame rate).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.providers.video.base import VideoProbeError, VideoProber
from app.schemas.generative_video import ClipExpectation, ClipValidationError, ClipValidationReport
from app.schemas.video import VideoProbe

FPS_RANGE = (8.0, 120.0)  # anything outside is not a usable clip whatever its claims
ASPECT_TOLERANCE = 0.03  # relative difference allowed between the asked and measured aspect ratio
CONTAINER_NAMES = {"mp4": ("mp4", "mov"), "avi": ("avi",), "webm": ("webm", "matroska"), "mov": ("mov",)}


def _ratio(aspect: str) -> float:
    w, h = aspect.split(":")
    return int(w) / int(h)


def container_name(probe: VideoProbe) -> str:
    """Our short name for what the parser reported ("mov,mp4,m4a,..." -> "mp4")."""
    names = probe.container.split(",")
    for short, aliases in CONTAINER_NAMES.items():
        if any(n in aliases for n in names):
            return short
    return names[0]


class ClipValidator:
    name = "clip-validator/1"

    def __init__(self, prober: VideoProber) -> None:
        self.prober = prober

    def validate(self, path: Path, checksum: str, expect: ClipExpectation) -> ClipValidationReport:
        errors: list[ClipValidationError] = []
        checks: list[str] = []

        def fail(code, field, message, expected=None, actual=None):
            errors.append(ClipValidationError(code=code, field=field, message=message,
                                              expected=None if expected is None else str(expected),
                                              actual=None if actual is None else str(actual)))

        def report(**measured) -> ClipValidationReport:
            return ClipValidationReport(valid=not errors, stage=expect.stage, checksum=checksum, errors=errors,
                                        checks=checks, validator=self.name, **measured)

        checks.append("file")
        if not path.is_file() or path.stat().st_size == 0:
            fail("empty_file", "file", "the clip file is missing or empty")
            return report()
        checks.append("checksum")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != checksum:
            fail("checksum_mismatch", "checksum", "the clip's bytes do not match the stored checksum", checksum, actual)
            return report()
        checks.append("container")
        try:
            probe = self.prober.probe(path)
        except (VideoProbeError, OSError, ValueError) as exc:
            fail("unreadable", "file", f"no container parser could read the clip: {exc}")
            return report()
        container = container_name(probe)
        video = probe.video
        measured = {"container": container, "duration": round(probe.duration, 3), "has_audio": bool(probe.audio),
                    "codec": video.codec if video else None, "width": video.width if video else None,
                    "height": video.height if video else None, "fps": video.fps if video else None}
        if expect.containers and container not in expect.containers:
            fail("unsupported_container", "container", "the clip's container is not supported",
                 ", ".join(expect.containers), container)
        checks.append("video_stream")
        if video is None:
            fail("missing_video_stream", "video", "the clip has no video stream")
            return report(**measured)
        checks.append("decode")
        problems = self.prober.decode_errors(path)
        if problems:
            fail("corrupted", "video", f"the clip does not decode cleanly: {'; '.join(problems)[:300]}")
        checks.append("codec")
        if expect.codecs and video.codec not in expect.codecs:
            fail("unsupported_codec", "codec", "the clip's video codec is not supported", ", ".join(expect.codecs),
                 video.codec)
        checks.append("duration")
        if probe.duration <= 0:
            fail("zero_duration", "duration", "the clip has no duration", expect.duration, probe.duration)
        elif expect.stage == "raw" and probe.duration < expect.duration - expect.duration_tolerance:
            # a raw clip may run longer (it is cut to the plan) but never shorter than planned
            fail("duration_mismatch", "duration", "the clip is shorter than planned", f">= {expect.duration:g}",
                 f"{probe.duration:.3f}")
        elif expect.stage == "normalized" and abs(probe.duration - expect.duration) > expect.duration_tolerance:
            fail("duration_mismatch", "duration", "the normalised clip does not have the planned length",
                 f"{expect.duration:g}", f"{probe.duration:.3f}")
        checks.append("resolution")
        if expect.width is not None and expect.height is not None:
            if (video.width, video.height) != (expect.width, expect.height):
                fail("resolution_mismatch", "resolution", "the clip is not at the platform resolution",
                     f"{expect.width}x{expect.height}", f"{video.width}x{video.height}")
        else:
            if video.width < expect.min_width or video.height < expect.min_height:
                fail("too_small", "resolution", "the clip is too small to show", f">= {expect.min_width}x"
                     f"{expect.min_height}", f"{video.width}x{video.height}")
            if abs(video.width / video.height - _ratio(expect.aspect_ratio)) > ASPECT_TOLERANCE * _ratio(
                    expect.aspect_ratio):
                fail("aspect_mismatch", "aspect_ratio", "the clip does not have the planned aspect ratio",
                     expect.aspect_ratio, f"{video.width}x{video.height}")
        checks.append("fps")
        if expect.fps is not None:
            if abs(video.fps - expect.fps) > 0.01:
                fail("fps_mismatch", "fps", "the clip is not at the platform frame rate", expect.fps, video.fps)
        elif not FPS_RANGE[0] <= video.fps <= FPS_RANGE[1]:
            fail("fps_out_of_range", "fps", "the clip's frame rate is not usable",
                 f"{FPS_RANGE[0]:g}-{FPS_RANGE[1]:g}", video.fps)
        checks.append("audio")
        if expect.audio == "required" and not probe.audio:
            fail("missing_audio", "audio", "the plan mixes the clip's sound but the clip has no audio stream")
        elif expect.audio == "absent" and probe.audio:
            fail("unexpected_audio", "audio", "a muted clip must not carry an audio stream")
        return report(**measured)
