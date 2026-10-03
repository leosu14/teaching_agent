"""Deterministic video validation.

- VideoPlanValidator checks a VideoPlan before anything is composed: unique, ordered, contiguous segments for known
  slides; durations; every segment and narration time equal to the PresentationTimeline (which is authoritative);
  image and audio references that resolve to the given IMAGE_ASSET and AUDIO_ASSET artifacts with matching
  checksums; narration inside its segment and not overlapping; subtitle timing; transitions; and a supported
  configuration. An invalid plan never reaches a composer.
- VideoValidator checks a composed file with a real container parser (the configured VideoProber): an MP4
  container by its magic bytes and the parser, the checksum, a non-zero duration matching the timeline within the
  configured tolerance, the expected resolution, frame rate and codecs, an audio stream, narration audible where
  the timeline has it, silence on silent stretches, and sampled frames that are not empty and change between
  slides.
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import ValidationError

from app.observability.scope import ExecutionScope
from app.providers.video.base import VideoProbeError, VideoProber
from app.schemas.events import EventType
from app.schemas.generative_video import IMPLEMENTED_STRATEGIES
from app.schemas.video import (
    IMAGE_MEDIA_TYPES,
    TIME_EPSILON,
    TransitionType,
    VideoPlan,
    VideoPlanIssue,
    VideoPlanValidationReport,
    VideoPlanValidationRequest,
    VideoValidationError,
    VideoValidationReport,
    VideoValidationRequest,
)
from app.tools.base import Tool, ToolError

PLAN_CHECKS = ["schema", "segments", "unique_segment_ids", "slide_refs", "ordering", "durations", "timeline",
               "total_duration", "image_refs", "audio_refs", "audio_overlap", "subtitles", "transitions", "config"]
NARRATION_FLOOR_DB = -45.0  # a narrated stretch must peak above this
SILENCE_CEILING_DB = -50.0  # a silent stretch must stay below this
FRAME_MIN_STDDEV = 0.5  # luma levels: below this a frame is a flat colour (empty)


def _same(a: float, b: float, eps: float = TIME_EPSILON) -> bool:
    return abs(a - b) <= eps


class VideoPlanInvalid(ToolError):
    def __init__(self, report: VideoPlanValidationReport) -> None:
        super().__init__("video plan failed validation: " + "; ".join(e.describe() for e in report.errors))
        self.report = report


class VideoPlanValidator:
    name = "video-plan-validator/1"

    def validate(self, request: VideoPlanValidationRequest) -> VideoPlanValidationReport:
        try:
            plan = VideoPlan.model_validate(request.plan)
        except ValidationError as exc:
            errors = [VideoPlanIssue(code="invalid_schema", field=".".join(str(p) for p in err["loc"]) or None,
                                     message=err["msg"]) for err in exc.errors()]
            return VideoPlanValidationReport(valid=False, video_plan_id=request.plan.get("video_plan_id"),
                                             errors=errors, checks=["schema"], validator=self.name)
        errors = self._check(plan, request)
        checks = [*PLAN_CHECKS, "generated_clips"] if plan.generated_clips() or request.generated_clips \
            else PLAN_CHECKS
        return VideoPlanValidationReport(valid=not errors, video_plan_id=plan.video_plan_id, errors=errors,
                                         checks=checks, validator=self.name, plan=plan)

    def _check(self, plan: VideoPlan, req: VideoPlanValidationRequest) -> list[VideoPlanIssue]:
        errors: list[VideoPlanIssue] = []

        def issue(code, message, segment_id=None, field=None) -> None:
            errors.append(VideoPlanIssue(code=code, message=message, segment_id=segment_id, field=field))

        timeline, config = req.timeline, plan.config
        # Configuration.
        if (plan.resolution.width, plan.resolution.height) != (config.width, config.height):
            issue("unsupported_config", "the plan's resolution differs from its configuration", field="resolution")
        if plan.fps != config.fps:
            issue("unsupported_config", "the plan's frame rate differs from its configuration", field="fps")
        if plan.timeline_ref != req.timeline_artifact_id or plan.timeline_id != timeline.timeline_id:
            issue("reference_mismatch", "the plan does not reference the given timeline", field="timeline_ref")
        if plan.presentation_ref != req.presentation_artifact_id or \
                timeline.presentation_artifact_id != req.presentation_artifact_id:
            issue("reference_mismatch", "the plan, timeline and presentation do not belong together",
                  field="presentation_ref")
        if not plan.slides:
            issue("no_segments", "the plan has no segments")
            return errors

        # Segments: unique, known slides, ordered and contiguous, timed exactly as the timeline.
        ids = [s.segment_id for s in plan.slides]
        for dup in sorted({i for i in ids if ids.count(i) > 1}):
            issue("duplicate_segment_id", "segment id is used more than once", dup)
        known = set(req.slide_ids)
        for s in plan.slides:
            if s.slide_id not in known:
                issue("unknown_slide", f"slide {s.slide_id} is not in the presentation", s.segment_id, "slide_id")
        if [s.order for s in plan.slides] != list(range(1, len(plan.slides) + 1)):
            issue("segment_order", "segments must be ordered 1..n", field="order")
        cursor = 0.0
        frame = 1 / plan.fps
        for s in plan.slides:
            if s.duration <= 0 or s.end_time < s.start_time:
                issue("negative_duration", "a segment needs a positive duration", s.segment_id, "duration")
            elif s.duration < frame:
                issue("negative_duration", "a segment is shorter than one frame", s.segment_id, "duration")
            if not _same(s.duration, s.end_time - s.start_time):
                issue("segment_order", "duration differs from end - start", s.segment_id, "duration")
            if not _same(s.start_time, cursor):
                issue("segment_order", f"segment starts at {s.start_time}s, not at {cursor}s (gap or overlap)",
                      s.segment_id, "start_time")
            cursor = s.end_time
        if [s.slide_id for s in plan.slides] != [t.slide_id for t in timeline.slides]:
            issue("timeline_mismatch", "segments do not follow the timeline's slides in order")
        else:
            for s, t in zip(plan.slides, timeline.slides):
                if not (_same(s.start_time, t.start_time) and _same(s.end_time, t.end_time)):
                    issue("timeline_mismatch", f"segment is timed {s.start_time}-{s.end_time}s but the timeline "
                          f"says {t.start_time}-{t.end_time}s", s.segment_id)
        if not (_same(plan.duration, timeline.duration) and _same(plan.duration, cursor)):
            issue("duration_mismatch", f"plan duration {plan.duration}s, segments end at {cursor}s, timeline is "
                  f"{timeline.duration}s", field="duration")

        # Images.
        images = {i.artifact_id: i for i in req.image_assets}
        for s in plan.slides:
            img = s.visual_ref.image
            if img is None:
                continue
            known_img = images.get(img.artifact_id)
            if known_img is None:
                issue("unknown_image", f"image {img.artifact_id} is not an IMAGE_ASSET of this lesson",
                      s.segment_id, "visual_ref.image")
            elif (known_img.checksum, known_img.uri) != (img.checksum, img.uri):
                issue("unknown_image", f"image {img.artifact_id} does not match its IMAGE_ASSET", s.segment_id,
                      "visual_ref.image")
            if img.media_type not in IMAGE_MEDIA_TYPES:
                issue("unsupported_image", f"{img.media_type} cannot be composed", s.segment_id, "visual_ref.image")

        # Narration: known AUDIO_ASSETs at exactly their timeline position, inside their segment, not overlapping.
        audio = {a.artifact_id: a for a in req.audio_assets}
        timed = {t.segment_id: t for t in timeline.segments}
        tracks = {t.track_id: t for t in plan.audio_tracks}
        if len(tracks) != len(plan.audio_tracks):
            issue("duplicate_segment_id", "an audio track id is used more than once", field="audio_tracks")
        referenced = [r for s in plan.slides for r in s.audio_refs]
        for s in plan.slides:
            for ref in s.audio_refs:
                t = tracks.get(ref)
                if t is None:
                    issue("unknown_reference", f"audio track {ref} does not exist", s.segment_id, "audio_refs")
                elif t.slide_id != s.slide_id or t.start_time < s.start_time - TIME_EPSILON or \
                        t.end_time > s.end_time + TIME_EPSILON:
                    issue("audio_outside_segment", f"audio track {ref} lies outside its segment", s.segment_id)
        for t in plan.audio_tracks:
            if referenced.count(t.track_id) != 1:
                issue("unknown_reference", f"audio track {t.track_id} must belong to exactly one segment",
                      field="audio_tracks")
            asset = audio.get(t.artifact_id)
            if asset is None or (asset.checksum, asset.uri) != (t.checksum, t.uri):
                issue("unknown_audio", f"audio {t.artifact_id} is not a matching AUDIO_ASSET", t.track_id)
                continue
            tt = timed.get(t.track_id)
            if tt is None or tt.audio_artifact_id != t.artifact_id or not (
                    _same(tt.start_time, t.start_time) and _same(tt.end_time, t.end_time)):
                issue("timeline_mismatch", "audio is not placed where the timeline puts it", t.track_id)
            if t.duration <= 0 or not _same(t.duration, t.end_time - t.start_time):
                issue("negative_duration", "audio track duration differs from end - start", t.track_id)
            if abs(asset.duration - t.duration) > config.duration_tolerance:
                issue("audio_duration_mismatch", f"the audio is {asset.duration:.3f}s but its slot is "
                      f"{t.duration:.3f}s", t.track_id)
        if set(timed) - set(tracks):
            issue("timeline_mismatch", f"timeline narration missing from the plan: {sorted(set(timed) - set(tracks))}")
        if not config.allow_audio_overlap:
            ordered = sorted(plan.audio_tracks, key=lambda t: t.start_time)
            for a, b in zip(ordered, ordered[1:]):
                if b.start_time < a.end_time - TIME_EPSILON:
                    issue("audio_overlap", f"{a.track_id} and {b.track_id} overlap", b.track_id)

        # Subtitles.
        if plan.subtitle_track is not None:
            subs = plan.subtitle_track.subtitles
            sub_ids = {c.subtitle_id for c in subs}
            if len(sub_ids) != len(subs):
                issue("subtitle_timing", "subtitle ids must be unique", field="subtitle_track")
            for c in subs:
                t = tracks.get(c.segment_ref)
                if t is None:
                    issue("unknown_reference", f"subtitle {c.subtitle_id} refers to unknown narration "
                          f"{c.segment_ref}", field="subtitle_track")
                elif c.start_time < t.start_time - TIME_EPSILON or c.end_time > t.end_time + TIME_EPSILON or \
                        c.end_time <= c.start_time:
                    issue("subtitle_timing", f"subtitle {c.subtitle_id} lies outside its narration", t.track_id)
            ordered_subs = sorted(subs, key=lambda c: c.start_time)
            for a, b in zip(ordered_subs, ordered_subs[1:]):
                if b.start_time < a.end_time - TIME_EPSILON:
                    issue("subtitle_overlap", f"subtitles {a.subtitle_id} and {b.subtitle_id} overlap")
            for s in plan.slides:
                for ref in s.subtitle_refs:
                    if ref not in sub_ids:
                        issue("unknown_reference", f"subtitle {ref} does not exist", s.segment_id, "subtitle_refs")
        elif any(s.subtitle_refs for s in plan.slides):
            issue("unknown_reference", "segments reference subtitles but the plan has no subtitle track")

        # Generated clips: known, matching GENERATED_VIDEO_ASSETs in the platform format, an implemented insertion
        # strategy, inside their segment and never longer than the clip itself.
        clips = {c.artifact_id: c for c in req.generated_clips}
        for s in plan.slides:
            clip = s.generated
            if clip is None:
                continue
            known_clip = clips.get(clip.artifact_id)
            if known_clip is None or (known_clip.checksum, known_clip.uri, known_clip.slide_id) != (
                    clip.checksum, clip.uri, s.slide_id):
                issue("unknown_generated_clip", f"clip {clip.artifact_id} is not a matching GENERATED_VIDEO_ASSET "
                      "for this slide", s.segment_id, "generated")
            if clip.strategy not in IMPLEMENTED_STRATEGIES:
                issue("unsupported_strategy", f"insertion strategy {clip.strategy.value} is not implemented",
                      s.segment_id, "generated.strategy")
            if (clip.width, clip.height) != (config.width, config.height) or abs(clip.fps - config.fps) > 0.01 or \
                    clip.media_type != config.media_type:
                issue("clip_format", "the clip is not normalised to the video's resolution, frame rate and "
                      "container", s.segment_id, "generated")
            if clip.audio == "mixed" and not clip.has_audio:
                issue("clip_format", "the clip's sound is mixed but the clip has no audio stream", s.segment_id,
                      "generated.audio")
            if clip.start_time < s.start_time - TIME_EPSILON or clip.end_time > s.end_time + TIME_EPSILON or \
                    clip.end_time <= clip.start_time:
                issue("clip_timing", "the clip lies outside its segment", s.segment_id, "generated")
            elif clip.duration > clip.asset_duration + TIME_EPSILON:
                issue("clip_timing", f"the clip plays for {clip.duration:.3f}s but is {clip.asset_duration:.3f}s "
                      "long", s.segment_id, "generated")

        # Transitions.
        if plan.slides[0].transition.type != TransitionType.CUT:
            issue("invalid_transition", "the first segment starts with a cut", plan.slides[0].segment_id)
        for prev, s in zip(plan.slides, plan.slides[1:]):
            tr = s.transition
            if tr.type == TransitionType.FADE and tr.duration > min(prev.duration, s.duration) + TIME_EPSILON:
                issue("invalid_transition", "a fade is longer than the segments it joins", s.segment_id)
        expected = [(a.segment_id, b.segment_id, b.transition) for a, b in zip(plan.slides, plan.slides[1:])]
        actual = [(t.from_segment, t.to_segment, t.transition) for t in plan.transitions]
        if expected != actual:
            issue("invalid_transition", "the transition list differs from the segments' transitions",
                  field="transitions")
        return errors


class VideoPlanValidationTool(Tool[VideoPlanValidationRequest, VideoPlanValidationReport]):
    name = "video_plan.validate"
    description = "Validate a VideoPlan against its PresentationTimeline and the IMAGE_ASSET and AUDIO_ASSET " \
                  "artifacts it references. With enforce, an invalid plan is an error."
    input_model = VideoPlanValidationRequest
    output_model = VideoPlanValidationReport

    def __init__(self, validator: VideoPlanValidator | None = None) -> None:
        self._validator = validator or VideoPlanValidator()

    async def run(self, data: VideoPlanValidationRequest, scope: ExecutionScope) -> VideoPlanValidationReport:
        report = self._validator.validate(data)
        scope.emit(EventType.VIDEO_PLAN_VALIDATED, tool=self.name, video_plan_id=report.video_plan_id,
                   valid=report.valid, errors=[e.describe() for e in report.errors][:20], enforce=data.enforce)
        if data.enforce and not report.valid:
            scope.emit(EventType.VIDEO_FAILED, tool=self.name, stage="plan_validation",
                       video_plan_id=report.video_plan_id, error="; ".join(e.describe() for e in report.errors)[:1000])
            raise VideoPlanInvalid(report)
        return report


# --- Output validation ------------------------------------------------------------------------


class VideoValidator:
    name = "video-validator/1"

    def __init__(self, prober: VideoProber) -> None:
        self.prober = prober

    def validate(self, req: VideoValidationRequest, path: Path, checksum: str) -> VideoValidationReport:
        errors: list[VideoValidationError] = []
        checks: list[str] = []
        config = req.config

        def fail(code, field, message, expected=None, actual=None) -> None:
            errors.append(VideoValidationError(code=code, field=field, message=message,
                                               expected=None if expected is None else str(expected),
                                               actual=None if actual is None else str(actual)))

        def report(probe=None, levels=None) -> VideoValidationReport:
            return VideoValidationReport(valid=not errors, errors=errors, checks=checks, probe=probe,
                                         audio_levels={k: (v if math.isfinite(v) else -200.0)
                                                       for k, v in (levels or {}).items()},
                                         validator=f"{self.name}+{self.prober.name}")

        checks.append("content")
        size = path.stat().st_size if path.exists() else 0
        if size == 0:
            fail("empty_file", "object", "the file is empty")
            return report()
        checks.append("checksum")
        if checksum != req.object.checksum or size != req.object.size_bytes:
            fail("checksum_mismatch", "object.checksum", "stored bytes do not match the recorded checksum",
                 req.object.checksum, checksum)
        checks.append("container")
        with path.open("rb") as f:
            head = f.read(12)
        if head[4:8] != b"ftyp":
            fail("not_mp4", "object", "the file is not an ISO base media (MP4) file", "ftyp box", head[4:8])
            return report()
        try:
            probe = self.prober.probe(path)
        except VideoProbeError as exc:
            fail("unreadable", "object", f"the container parser cannot read the file: {exc}")
            return report()
        if "mp4" not in probe.container.split(","):
            fail("not_mp4", "container", "the parser does not see an MP4 container", "mp4", probe.container)

        checks.append("duration")
        if probe.duration <= 0:
            fail("zero_duration", "duration", "the video has no duration", "> 0", probe.duration)
        elif abs(probe.duration - req.expected_duration) > config.duration_tolerance:
            fail("duration_mismatch", "duration", f"differs from the timeline by more than "
                 f"{config.duration_tolerance}s", round(req.expected_duration, 3), round(probe.duration, 3))

        checks.append("video_stream")
        v = probe.video
        if v is None:
            fail("missing_video_stream", "video", "the file has no video stream")
        else:
            if (v.width, v.height) != (config.width, config.height):
                fail("resolution_mismatch", "video.resolution", "unexpected resolution",
                     f"{config.width}x{config.height}", f"{v.width}x{v.height}")
            if abs(v.fps - config.fps) > 0.01:
                fail("fps_mismatch", "video.fps", "unexpected frame rate", config.fps, round(v.fps, 3))
            if v.codec != config.codec.value:
                fail("codec_mismatch", "video.codec", "unexpected video codec", config.codec.value, v.codec)

        checks.append("audio_stream")
        if not probe.audio:
            fail("missing_audio_stream", "audio", "the file has no audio stream")
            return report(probe)
        if probe.audio[0].codec != config.audio_codec.value:
            fail("audio_codec_mismatch", "audio.codec", "unexpected audio codec", config.audio_codec.value,
                 probe.audio[0].codec)

        levels: dict[str, float] = {}
        if req.audio_windows:
            checks.append("narration")
            try:
                levels = self.prober.audio_levels(path, req.audio_windows)
            except VideoProbeError as exc:
                fail("unreadable", "audio", f"the audio cannot be decoded: {exc}")
            for w in req.audio_windows:
                level = levels.get(w.label, float("-inf"))
                if w.expect_sound and level < NARRATION_FLOOR_DB:
                    fail("missing_narration", f"audio.{w.label}", f"no narration between {w.start_time:.2f}s and "
                         f"{w.end_time:.2f}s", f">= {NARRATION_FLOOR_DB} dBFS", f"{level:.1f} dBFS")
                if not w.expect_sound and level > SILENCE_CEILING_DB:
                    fail("unexpected_sound", f"audio.{w.label}", f"sound where the timeline has silence "
                         f"({w.start_time:.2f}s-{w.end_time:.2f}s)", f"<= {SILENCE_CEILING_DB} dBFS",
                         f"{level:.1f} dBFS")

        if req.frame_samples:
            checks.append("frames")
            try:
                stats = self.prober.frame_stats(path, req.frame_samples)
            except VideoProbeError as exc:
                fail("unreadable", "frames", f"frames cannot be decoded: {exc}")
                return report(probe, levels)
            previous = None
            for sample in req.frame_samples:
                st = stats[sample.label]
                if st.stddev < FRAME_MIN_STDDEV:
                    fail("empty_frame", f"frames.{sample.label}", f"the frame at {sample.time:.2f}s is empty")
                if previous is not None and sample.expect_change and not st.differs_from(previous):
                    fail("no_transition", f"frames.{sample.label}", f"the picture does not change at "
                         f"{sample.label}")
                previous = st
        return report(probe, levels)
