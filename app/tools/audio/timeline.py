"""Timing and the PRESENTATION_TIMELINE artifact.

The resolver is deterministic and uses the measured duration of every generated AUDIO_ASSET, never the planning
estimate. Slides play in deck order; each segment starts after its pause_before and is followed by its
pause_after; a slide without narration is held for `silent_slide_seconds`. Times are resolved in whole
milliseconds so they add up exactly. The timeline references the PRESENTATION artifact and slide ids, not the
PPTX file, and is stored as JSON linked to the presentation, the audio plan and every audio asset.
"""

from __future__ import annotations

import hashlib
import json

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.schemas.artifact import ArtifactType
from app.schemas.audio import (
    AudioPlan,
    AudioPlanValidationRequest,
    PresentationTimeline,
    SegmentTiming,
    SlideTiming,
    TimelineRequest,
    TimelineResult,
)
from app.schemas.events import EventType
from app.tools.audio.validation import AudioPlanInvalid, AudioPlanValidator
from app.tools.base import Tool, ToolError

RESOLVER = "sequential-timing/1"


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _s(ms: int) -> float:
    return ms / 1000


def resolve_timing(plan: AudioPlan, durations: dict[str, float], slide_ids: list[str],
                   silent_slide_seconds: float = 3.0) -> tuple[AudioPlan, list[SlideTiming]]:
    """The plan with start/end/duration on every voiced segment, and one SlideTiming per slide. Segments without a
    measured duration (not voiced) get no timing and take no time."""
    cursor = 0
    timed: dict[str, dict] = {}
    slides: list[SlideTiming] = []
    for order, slide_id in enumerate(slide_ids, start=1):
        start = cursor
        voiced = sorted((s for s in plan.for_slide(slide_id) if s.segment_id in durations), key=lambda s: s.order)
        if not voiced:
            cursor += _ms(silent_slide_seconds)
        for seg in voiced:
            cursor += _ms(seg.pause_before)
            length = max(1, _ms(durations[seg.segment_id]))
            timed[seg.segment_id] = {"start_time": _s(cursor), "end_time": _s(cursor + length),
                                     "duration": _s(length)}
            cursor += length + _ms(seg.pause_after)
        slides.append(SlideTiming(slide_id=slide_id, order=order, start_time=_s(start), end_time=_s(cursor),
                                  duration=_s(cursor - start), audio_segment_refs=[s.segment_id for s in voiced]))
    segments = [s.model_copy(update=timed.get(s.segment_id, {"start_time": None, "end_time": None, "duration": None}))
                for s in plan.segments]
    return plan.model_copy(update={"segments": segments}), slides


class PresentationTimelineTool(Tool[TimelineRequest, TimelineResult]):
    name = "audio.timeline"
    description = "Resolve slide and segment timing from the measured audio durations and store the " \
                  "PRESENTATION_TIMELINE artifact linking the presentation, its slides and the audio assets."
    input_model = TimelineRequest
    output_model = TimelineResult
    permissions = frozenset({"artifact:write"})

    def __init__(self, artifacts: ArtifactService, validator: AudioPlanValidator | None = None) -> None:
        self._artifacts = artifacts
        self._validator = validator or AudioPlanValidator()

    async def run(self, data: TimelineRequest, scope: ExecutionScope) -> TimelineResult:
        if scope.task_id is None:
            raise ToolError("timelines can only be created inside a task")
        plan, narration = data.plan, data.narration
        try:
            result = self._resolve(data, scope.task_id, scope)
        except Exception as exc:
            scope.emit(EventType.AUDIO_FAILED, tool=self.name, stage="timeline", audio_plan_id=plan.audio_plan_id,
                       error=f"{type(exc).__name__}: {exc}"[:1000])
            raise
        timeline, artifact = result.timeline, result.artifact
        scope.emit(EventType.TIMELINE_CREATED, tool=self.name, timeline_id=timeline.timeline_id,
                   artifact_id=artifact.artifact_id, version=artifact.version,
                   presentation_artifact_id=timeline.presentation_artifact_id, slides=len(timeline.slides),
                   segments=len(timeline.segments), duration=timeline.duration,
                   missing_segments=timeline.missing_segments)
        scope.emit(EventType.AUDIO_COMPLETED, tool=self.name, audio_plan_id=plan.audio_plan_id,
                   status=narration.status, assets=len(narration.assets), failures=len(narration.failures),
                   duration=timeline.duration, timeline_artifact_id=artifact.artifact_id)
        return result

    def _resolve(self, data: TimelineRequest, task_id: str, scope: ExecutionScope) -> TimelineResult:
        plan, narration = data.plan, data.narration
        durations = narration.durations()
        timed, slides = resolve_timing(plan, durations, data.slide_ids, data.silent_slide_seconds)
        report = self._validator.validate(AudioPlanValidationRequest(
            plan=timed.model_dump(mode="json"), slide_ids=data.slide_ids, check_voices=False), None)
        if not report.valid:
            raise AudioPlanInvalid(report)
        artifacts = {a.segment_id: a.artifact_id for a in narration.assets}
        segments = [SegmentTiming(segment_id=s.segment_id, slide_id=s.slide_id, order=s.order,
                                  audio_artifact_id=artifacts[s.segment_id], start_time=s.start_time,
                                  end_time=s.end_time, duration=s.duration)
                    for s in timed.segments if s.segment_id in durations]
        missing = [s.segment_id for s in plan.segments if s.segment_id not in durations]
        identity = json.dumps({"plan": plan.audio_plan_id, "presentation": data.presentation_artifact_id,
                               "slides": [s.model_dump(mode="json") for s in slides],
                               "segments": [s.model_dump(mode="json") for s in segments]}, sort_keys=True)
        timeline = PresentationTimeline(
            timeline_id="tl_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            presentation_artifact_id=data.presentation_artifact_id, deck_id=plan.deck_id,
            audio_plan_id=plan.audio_plan_id, audio_plan_artifact_id=data.audio_plan_artifact_id,
            language=plan.language, voice=plan.voice, duration=slides[-1].end_time, slides=slides, segments=segments,
            missing_segments=missing, resolver=RESOLVER,
            metadata={"silent_slide_seconds": data.silent_slide_seconds, "narration_status": narration.status},
        )
        artifact = self._artifacts.store(
            task_id=task_id, name=data.name, type=ArtifactType.PRESENTATION_TIMELINE, media_type="application/json",
            content=timeline.model_dump_json(indent=2).encode("utf-8"), provider="teaching-agent",
            parent_ids=[data.presentation_artifact_id, data.audio_plan_artifact_id, *timeline.audio_artifact_ids()],
            metadata={"timeline_id": timeline.timeline_id, "deck_id": timeline.deck_id,
                      "presentation_artifact_id": timeline.presentation_artifact_id,
                      "audio_plan_id": timeline.audio_plan_id, "duration": timeline.duration,
                      "slides": len(slides), "segments": len(segments), "missing_segments": missing,
                      "audio_artifact_ids": timeline.audio_artifact_ids()},
            scope=scope,
        )
        return TimelineResult(timeline=timeline, plan=timed, artifact=artifact)
