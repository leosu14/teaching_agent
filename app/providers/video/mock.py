"""Deterministic mock composer and prober, for tests that should not encode video.

The mock "MP4" is an ISO base media file in shape only: an `ftyp` box followed by a `free` box holding a JSON
manifest of what a real composer would have encoded (frames, streams, where narration and subtitles are). The
mock prober reads that manifest back and refuses anything else, so validation logic runs unchanged against it.
Same plan, same bytes.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from app.providers.video.base import (
    FrameStats,
    MediaReader,
    VideoComposer,
    VideoCompositionError,
    VideoProbeError,
    VideoProber,
    read_media,
)
from app.providers.video.ffmpeg import frame_at
from app.schemas.video import (
    AudioStreamProbe,
    AudioWindow,
    ComposedVideo,
    FrameSample,
    VideoPlan,
    VideoProbe,
    VideoStreamProbe,
)

MANIFEST_KIND = "teaching-agent/mock-mp4"


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


class MockVideoComposer(VideoComposer):
    name = "mock-composer/1"

    def __init__(self, media: MediaReader | None = None, *, fail: bool = False) -> None:
        self._media = media
        self.fail = fail
        self.calls = 0

    def compose(self, plan: VideoPlan, workspace: Path) -> ComposedVideo:
        self.calls += 1
        if self.fail:
            raise VideoCompositionError("mock composer configured to fail")
        if self._media is not None:  # the inputs must exist and match their checksums, as for a real composer
            for seg in plan.slides:
                if seg.visual_ref.image is not None:
                    read_media(self._media, seg.visual_ref.image.uri, seg.visual_ref.image.checksum, "image")
            for track in plan.audio_tracks:
                read_media(self._media, track.uri, track.checksum, f"audio {track.artifact_id}")
        frames = frame_at(plan.duration, plan.fps)
        cues = plan.subtitle_track.subtitles if plan.subtitle_track and plan.config.subtitles.enabled else []
        manifest = {
            "kind": MANIFEST_KIND, "video_plan_id": plan.video_plan_id, "width": plan.resolution.width,
            "height": plan.resolution.height, "fps": plan.fps, "frames": frames, "duration": frames / plan.fps,
            "codec": plan.config.codec.value, "audio_codec": plan.config.audio_codec.value,
            "sample_rate": plan.config.audio_sample_rate, "channels": plan.config.audio_channels,
            "audio": [[t.start_time, t.end_time] for t in plan.audio_tracks],
            "segments": [[s.start_time, s.end_time, s.visual_ref.card.title,
                          s.visual_ref.image.checksum if s.visual_ref.image else None] for s in plan.slides],
            "subtitles": [[c.start_time, c.end_time, c.text] for c in cues],
        }
        payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
        out = workspace / "video.mp4"
        out.write_bytes(_box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41") + _box(b"free", payload))
        return ComposedVideo(path=str(out), media_type=plan.config.media_type, composer=self.name, frames=frames,
                             duration=frames / plan.fps, render_seconds=0.0, cpu_seconds=0.0,
                             subtitles_burned=len(cues), metadata={"mock": True})


def read_manifest(path: Path) -> dict:
    data = path.read_bytes()
    pos, manifest = 0, None
    if data[4:8] != b"ftyp":
        raise VideoProbeError("not an ISO base media file")
    while pos + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if size < 8:
            raise VideoProbeError("corrupt box")
        if kind == b"free":
            try:
                manifest = json.loads(data[pos + 8:pos + size])
            except json.JSONDecodeError as exc:
                raise VideoProbeError("unreadable mock manifest") from exc
        pos += size
    if not isinstance(manifest, dict) or manifest.get("kind") != MANIFEST_KIND:
        raise VideoProbeError("no mock video manifest")
    return manifest


class MockVideoProber(VideoProber):
    name = "mock-prober/1"

    def probe(self, path: Path) -> VideoProbe:
        m = read_manifest(path)
        return VideoProbe(
            container="mov,mp4,m4a,3gp,3g2,mj2", brand="isom", duration=m["duration"],
            size_bytes=path.stat().st_size,
            video=VideoStreamProbe(codec=m["codec"], width=m["width"], height=m["height"], fps=float(m["fps"]),
                                   pixel_format="yuv420p", frames=m["frames"], duration=m["duration"]),
            audio=[AudioStreamProbe(codec=m["audio_codec"], sample_rate=m["sample_rate"], channels=m["channels"],
                                    duration=m["duration"])],
            prober=self.name,
        )

    def audio_levels(self, path: Path, windows: list[AudioWindow]) -> dict[str, float]:
        spans = read_manifest(path)["audio"]
        return {w.label: -6.0 if any(a < w.end_time and w.start_time < b for a, b in spans) else float("-inf")
                for w in windows}

    def frame_stats(self, path: Path, samples: list[FrameSample]) -> dict[str, FrameStats]:
        m = read_manifest(path)
        out = {}
        for s in samples:
            seg = next((x for x in m["segments"] if x[0] <= s.time < x[1]), m["segments"][-1])
            cue = next((c[2] for c in m["subtitles"] if c[0] <= s.time < c[1]), "")
            thumb = hashlib.sha256(json.dumps([seg, cue]).encode()).digest() * 72
            mean = sum(thumb) / len(thumb)
            out[s.label] = FrameStats(mean, 1.0 + mean / 255, thumb)
        return out
