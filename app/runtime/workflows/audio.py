"""The narration stage and the audio failure policy.

NarrationNode voices a validated AudioPlan one segment at a time, through the ToolManager:
reuse an existing AUDIO_ASSET made from the same inputs whose bytes still match (audio.find_asset), otherwise
synthesize (tts.synthesize), validate and store (audio.create_asset). A segment that cannot be voiced is recorded
as a failure with its stage and reason, never skipped silently. Assets are durable as soon as they are stored, so
a node interrupted mid-plan resumes by reusing them instead of synthesizing them again.

Audio failure policy, applied by the workflow after the node:
- An optional segment that fails never stops the task: it becomes a warning and its slide plays without it.
- A required segment that fails follows `AudioFailurePolicy`: `fail` (the default) fails the task with an explicit
  error; `continue` completes it with a warning. Every audio asset already generated is kept either way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

from app.runtime.workflow.nodes import Node, NodeFatal, NodeResult, NodeRuntime, StateView
from app.schemas.artifact import Artifact
from app.schemas.audio import (
    AudioAssetLookup,
    AudioAssetLookupResult,
    AudioAssetRef,
    AudioAssetRequest,
    AudioSegment,
    AudioSegmentFailure,
    NarrationRequest,
    NarrationResult,
    TTSResult,
)
from app.schemas.events import EventType
from app.tools.base import ToolCaller, ToolError, ToolNotFound, ToolPermissionError

AudioFailurePolicy = Literal["fail", "continue"]

NARRATION_TOOLS = frozenset({"audio.find_asset", "tts.synthesize", "audio.create_asset"})
NARRATION_PERMISSIONS = frozenset({"media:generate", "artifact:write", "artifact:read"})


class AudioRequired(NodeFatal):
    """A required narration segment could not be voiced and the workflow requires it."""


def asset_name(segment: AudioSegment) -> str:
    return f"audio_{segment.segment_id}"


@dataclass(frozen=True, kw_only=True)
class NarrationNode(Node):
    build_input: Callable[[StateView], NarrationRequest]
    kind: ClassVar[str] = "narration"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        data = self.build_input(rt.view)
        caller = ToolCaller(caller_id=f"node:{self.id}", allowed_tools=NARRATION_TOOLS,
                            permissions=NARRATION_PERMISSIONS)
        assets: list[AudioAssetRef] = []
        artifacts: dict[str, Artifact] = {}
        failures: list[AudioSegmentFailure] = []
        for segment in data.plan.segments:
            outcome = await self._voice(segment, data, caller, rt)
            if isinstance(outcome, AudioSegmentFailure):
                failures.append(outcome)
            else:
                ref, artifact = outcome
                assets.append(ref)
                artifacts.setdefault(artifact.artifact_id, artifact)

        warnings = [f"{'Required' if f.required else 'Optional'} narration {f.segment_id} for slide {f.slide_id} "
                    f"was not generated ({f.stage}): {f.reason}" for f in failures]
        status = "failed" if any(f.required for f in failures) else "partial" if failures else "complete"
        result = NarrationResult(status=status, audio_plan_id=data.plan.audio_plan_id, assets=assets,
                                 artifacts=list(artifacts.values()), failures=failures, warnings=warnings)
        if status == "failed":
            rt.scope.emit(EventType.AUDIO_FAILED, stage="tts", audio_plan_id=data.plan.audio_plan_id,
                          assets=len(assets), failures=[f.model_dump(mode="json", exclude={"errors"})
                                                        for f in failures])
        return NodeResult(output=result)

    async def _voice(self, segment: AudioSegment, data: NarrationRequest, caller: ToolCaller,
                     rt: NodeRuntime) -> tuple[AudioAssetRef, Artifact] | AudioSegmentFailure:
        scope = rt.scope
        request = segment.tts_request(data.output_format, data.sample_rate)
        name = asset_name(segment)

        def failure(stage, reason: str, errors=()) -> AudioSegmentFailure:
            return AudioSegmentFailure(segment_id=segment.segment_id, slide_id=segment.slide_id,
                                       required=segment.required, stage=stage, reason=reason[:1000],
                                       errors=list(errors))

        found = await rt.tools.call(caller, "audio.find_asset",
                                    AudioAssetLookup(name=name, input_hash=request.fingerprint()), scope)
        assert isinstance(found, AudioAssetLookupResult)
        if found.artifact is not None:
            return self._created(segment, found.artifact, scope, reused=True)

        scope.emit(EventType.TTS_STARTED, segment_id=segment.segment_id, slide_id=segment.slide_id,
                   language=segment.language, voice=segment.voice, characters=len(segment.text),
                   format=data.output_format)
        try:
            tts = await rt.tools.call(caller, "tts.synthesize", request, scope)
        except (ToolPermissionError, ToolNotFound):
            raise
        except ToolError as exc:
            scope.emit(EventType.TTS_COMPLETED, segment_id=segment.segment_id, ok=False, error=str(exc)[:1000])
            return failure("tts", str(exc))
        assert isinstance(tts, TTSResult)
        scope.emit(EventType.TTS_COMPLETED, segment_id=segment.segment_id, ok=True, provider=tts.provider,
                   model=tts.model, duration=tts.duration, checksum=tts.audio.checksum,
                   reused_object=tts.audio.reused, usage=tts.usage.model_dump(),
                   ignored_parameters=tts.ignored_parameters)

        try:
            artifact = await rt.tools.call(caller, "audio.create_asset", AudioAssetRequest(
                name=name, segment=segment, audio_plan_id=data.plan.audio_plan_id, tts=tts,
                expected_sample_rate=data.sample_rate, parent_ids=[data.audio_plan_artifact_id]), scope)
        except (ToolPermissionError, ToolNotFound):
            raise
        except ToolError as exc:
            report = getattr(exc, "report", None)
            if report is None:
                return failure("store", str(exc))
            scope.emit(EventType.AUDIO_VALIDATION_FAILED, segment_id=segment.segment_id,
                       checksum=tts.audio.checksum, errors=[e.model_dump(exclude_none=True) for e in report.errors])
            codes = sorted({e.code for e in report.errors})
            return failure("validate", f"audio failed validation: {', '.join(codes)}", report.errors)
        assert isinstance(artifact, Artifact)
        return self._created(segment, artifact, scope, reused=False)

    @staticmethod
    def _created(segment: AudioSegment, artifact: Artifact, scope, *, reused: bool) -> tuple[AudioAssetRef, Artifact]:
        meta = artifact.metadata
        ref = AudioAssetRef(segment_id=segment.segment_id, slide_id=segment.slide_id, artifact_id=artifact.artifact_id,
                            asset_id=meta["asset_id"], checksum=artifact.content_hash, uri=artifact.uri,
                            media_type=artifact.media_type, duration=meta["duration"], reused=reused)
        scope.emit(EventType.AUDIO_ASSET_CREATED, segment_id=segment.segment_id, slide_id=segment.slide_id,
                   artifact_id=artifact.artifact_id, version=artifact.version, asset_id=ref.asset_id,
                   checksum=ref.checksum, duration=ref.duration, media_type=ref.media_type, reused=reused)
        return ref, artifact


def apply_audio_policy(result: NarrationResult, policy: AudioFailurePolicy) -> NarrationResult:
    required = [f for f in result.failures if f.required]
    if not required:
        return result
    reasons = "; ".join(f"{f.segment_id} (slide {f.slide_id}, {f.stage}): {f.reason}" for f in required)
    if policy == "fail":
        raise AudioRequired(f"required narration failed: {reasons}")
    warning = (f"Required narration failed and the audio policy is 'continue', so these slides play without it: "
               f"{reasons}")
    return result.model_copy(update={"warnings": [*result.warnings, warning]})
