"""Deterministic audio validation.

- AudioValidator checks generated audio against its own bytes: MIME type, a readable container, non-empty audio,
  a non-zero measured duration, the expected sample rate and channels, and the checksum. Declared values are
  compared with what the container parser measured, never trusted.
- AudioPlanValidator checks an AudioPlan: unique segment ids, known slides in deck order, contiguous segment
  order, non-empty and concise text, no citation metadata read aloud, supported language and voice, and (once
  timed) no negative durations and no overlapping segments.
"""

from __future__ import annotations

import hashlib

from pydantic import ValidationError

from app.observability.scope import ExecutionScope
from app.providers.tts.base import TTSProvider, TTSProviderError
from app.schemas.audio import (
    AUDIO_MEDIA_TYPES,
    AudioPlan,
    AudioPlanIssue,
    AudioPlanValidationReport,
    AudioPlanValidationRequest,
    AudioValidationError,
    AudioValidationReport,
    AudioValidationRequest,
    MeasuredAudio,
    Voice,
)
from app.schemas.events import EventType
from app.tools.base import Tool, ToolError, ToolTransientError
from app.utils.audio import AudioProbeError, probe_audio

DURATION_TOLERANCE = 0.01  # seconds between declared and measured duration
ALLOWED_MEDIA_TYPES = frozenset({*AUDIO_MEDIA_TYPES.values(), "audio/x-wav", "audio/wave"})


class AudioValidator:
    name = "audio-validator/1"

    def validate(self, req: AudioValidationRequest, content: bytes) -> AudioValidationReport:
        errors: list[AudioValidationError] = []
        checks: list[str] = []

        def fail(code, field, message, expected=None, actual=None) -> None:
            errors.append(AudioValidationError(code=code, field=field, message=message,
                                               expected=None if expected is None else str(expected),
                                               actual=None if actual is None else str(actual)))

        def report(measured: MeasuredAudio | None = None) -> AudioValidationReport:
            return AudioValidationReport(valid=not errors, errors=errors, checks=checks, measured=measured,
                                         validator=self.name)

        checks.append("content")
        if not content:
            fail("empty_audio", "object", "the audio has no bytes")
            return report()
        checks.append("checksum")
        digest = hashlib.sha256(content).hexdigest()
        if digest != req.object.checksum or len(content) != req.object.size_bytes:
            fail("checksum_mismatch", "object.checksum", "stored bytes do not match the recorded checksum",
                 req.object.checksum, digest)
        checks.append("media_type")
        if req.object.media_type not in ALLOWED_MEDIA_TYPES:
            fail("unsupported_media_type", "object.media_type", "not an audio MIME type",
                 sorted(ALLOWED_MEDIA_TYPES), req.object.media_type)
        checks.append("container")
        try:
            probe = probe_audio(content)
        except AudioProbeError as exc:
            fail("unreadable_audio", "object", f"the bytes are not readable audio: {exc}")
            return report()
        measured = MeasuredAudio(format=probe.format, media_type=probe.media_type, duration=probe.duration,
                                 sample_rate=probe.sample_rate, channels=probe.channels,
                                 sample_width=probe.sample_width, frames=probe.frames, checksum=digest,
                                 size_bytes=len(content))
        if probe.format != req.declared_format:
            fail("format_mismatch", "format", "the audio container differs from the declared format",
                 req.declared_format, probe.format)
        if probe.media_type != req.object.media_type:
            fail("format_mismatch", "object.media_type", "the stored MIME type differs from the audio container",
                 probe.media_type, req.object.media_type)
        checks.append("duration")
        if probe.frames == 0 or probe.duration <= 0:
            fail("zero_duration", "duration", "the audio has no samples", "> 0", probe.duration)
        elif abs(probe.duration - req.declared_duration) > DURATION_TOLERANCE:
            fail("duration_mismatch", "duration", "the measured duration differs from the declared one",
                 round(req.declared_duration, 4), round(probe.duration, 4))
        checks.append("sample_rate")
        for expected in {req.declared_sample_rate, req.expected_sample_rate} - {None}:
            if probe.sample_rate != expected:
                fail("sample_rate_mismatch", "sample_rate", "unexpected sample rate", expected, probe.sample_rate)
        checks.append("channels")
        for expected in {req.declared_channels, req.expected_channels} - {None}:
            if probe.channels != expected:
                fail("channel_mismatch", "channels", "unexpected channel count", expected, probe.channels)
        return report(measured)


PLAN_CHECKS = ["schema", "segments", "unique_segment_ids", "slide_refs", "ordering", "text", "unspoken_metadata"]


class AudioPlanInvalid(ToolError):
    """The audio plan failed validation. `report` holds the structured errors."""

    def __init__(self, report: AudioPlanValidationReport) -> None:
        super().__init__("audio plan failed validation: " + "; ".join(e.describe() for e in report.errors))
        self.report = report


class AudioPlanValidator:
    name = "audio-plan-validator/1"

    def validate(self, request: AudioPlanValidationRequest, voices: list[Voice] | None) -> AudioPlanValidationReport:
        try:
            plan = AudioPlan.model_validate(request.plan)
        except ValidationError as exc:
            errors = [AudioPlanIssue(code="invalid_schema", field=".".join(str(p) for p in err["loc"]) or None,
                                     message=err["msg"]) for err in exc.errors()]
            return AudioPlanValidationReport(valid=False, audio_plan_id=request.plan.get("audio_plan_id"),
                                             errors=errors, checks=["schema"], validator=self.name)
        checks = [*PLAN_CHECKS, *(["language", "voice"] if voices is not None else [])]
        timed = any(s.start_time is not None or s.end_time is not None or s.duration is not None
                    for s in plan.segments)
        if timed:
            checks.append("timing")
        errors = self._check(plan, request, voices, timed)
        return AudioPlanValidationReport(valid=not errors, audio_plan_id=plan.audio_plan_id, errors=errors,
                                         checks=checks, validator=self.name, plan=plan)

    def _check(self, plan: AudioPlan, req: AudioPlanValidationRequest, voices: list[Voice] | None,
               timed: bool) -> list[AudioPlanIssue]:
        errors: list[AudioPlanIssue] = []

        def issue(code, message, segment=None, field=None) -> None:
            errors.append(AudioPlanIssue(code=code, message=message, field=field,
                                         segment_id=segment.segment_id if segment else None))

        if not plan.segments:
            issue("no_segments", "the plan narrates nothing")
            return errors
        seen: set[str] = set()
        for seg in plan.segments:
            if seg.segment_id in seen:
                issue("duplicate_segment_id", f"segment id {seg.segment_id} is used more than once", seg)
            seen.add(seg.segment_id)

        position = {sid: i for i, sid in enumerate(req.slide_ids)}
        if [s.order for s in plan.segments] != list(range(1, len(plan.segments) + 1)):
            issue("segment_order", "segments must be listed in order 1..n", field="order")
        last = -1
        for seg in plan.segments:
            if seg.slide_id not in position:
                issue("unknown_slide", f"slide {seg.slide_id} is not in the presentation", seg, "slide_id")
                continue
            if position[seg.slide_id] < last:
                issue("slide_order", f"slide {seg.slide_id} is narrated after a later slide", seg, "slide_id")
            last = max(last, position[seg.slide_id])

        terms = [t.lower() for t in req.unspoken_terms]
        for seg in plan.segments:
            text = seg.text.strip()
            if not text:
                issue("empty_text", "a segment needs text to speak", seg, "text")
                continue
            words = len(text.split())
            if req.max_words_per_segment and words > req.max_words_per_segment:
                issue("too_long", f"{words} words; at most {req.max_words_per_segment}", seg, "text")
            lowered = text.lower()
            read = [t for t in terms if t in lowered]
            if read or "http://" in lowered or "https://" in lowered:
                issue("spoken_metadata", f"citation metadata would be read aloud: {(read or ['a URL'])[0]}", seg,
                      "text")

        if voices is not None:
            by_id = {v.voice_id: v for v in voices}
            for lang, where in ((plan.language, None), *((s.language, s) for s in plan.segments)):
                if not any(v.supports(lang) for v in voices):
                    issue("unsupported_language", f"no voice speaks {lang}", where, "language")
            for voice_id, lang, where in ((plan.voice, plan.language, None),
                                          *((s.voice, s.language, s) for s in plan.segments)):
                voice = by_id.get(voice_id)
                if voice is None:
                    issue("unknown_voice", f"voice {voice_id} is not in the provider's catalog", where, "voice")
                elif not voice.supports(lang):
                    issue("voice_language_mismatch", f"voice {voice_id} speaks {voice.language}, not {lang}", where,
                          "voice")
        if timed:
            self._timing(plan, issue)
        return errors

    @staticmethod
    def _timing(plan: AudioPlan, issue) -> None:
        voiced = []
        for seg in plan.segments:
            values = (seg.start_time, seg.end_time, seg.duration)
            if all(v is None for v in values):
                continue  # not voiced (an optional segment that failed)
            if any(v is None for v in values):
                issue("incomplete_timing", "start_time, end_time and duration are set together", seg, "timing")
                continue
            if seg.start_time < 0 or seg.duration <= 0 or seg.end_time < seg.start_time:
                issue("negative_duration", f"invalid interval {seg.start_time} -> {seg.end_time} "
                      f"({seg.duration}s)", seg, "timing")
                continue
            if abs(seg.end_time - seg.start_time - seg.duration) > 1e-6:
                issue("timing_mismatch", "duration differs from end_time - start_time", seg, "timing")
            voiced.append(seg)
        for prev, seg in zip(voiced, voiced[1:]):
            if seg.start_time < prev.end_time - 1e-9:
                issue("overlapping_timing", f"starts at {seg.start_time}, before {prev.segment_id} ends at "
                      f"{prev.end_time}", seg, "start_time")


class AudioPlanValidationTool(Tool[AudioPlanValidationRequest, AudioPlanValidationReport]):
    name = "audio_plan.validate"
    description = "Validate an audio plan deterministically: segment ids and ordering, slide references, text, " \
                  "citation metadata, language and voice against the TTS provider's catalog, and timing."
    input_model = AudioPlanValidationRequest
    output_model = AudioPlanValidationReport

    def __init__(self, provider: TTSProvider, validator: AudioPlanValidator | None = None) -> None:
        self._provider = provider
        self._validator = validator or AudioPlanValidator()

    async def run(self, data: AudioPlanValidationRequest, scope: ExecutionScope) -> AudioPlanValidationReport:
        voices = None
        if data.check_voices:
            try:
                voices = await self._provider.voices()
            except (TTSProviderError, ConnectionError, OSError) as exc:
                raise ToolTransientError(f"TTS provider '{self._provider.name}' voice list failed: {exc}") from exc
        report = self._validator.validate(data, voices)
        if not data.enforce:
            return report
        if not report.valid:
            scope.emit(EventType.AUDIO_FAILED, tool=self.name, stage="validation", audio_plan_id=report.audio_plan_id,
                       errors=[e.model_dump(exclude_none=True) for e in report.errors])
            raise AudioPlanInvalid(report)
        assert report.plan is not None
        scope.emit(EventType.AUDIO_PLAN_VALIDATED, tool=self.name, audio_plan_id=report.audio_plan_id,
                   segments=len(report.plan.segments), checks=report.checks)
        return report
