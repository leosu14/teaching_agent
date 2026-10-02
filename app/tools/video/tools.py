"""Video tools: the workflow's only way to compose, validate and store a video. Each is a thin wrapper over
VideoService that emits the video events and records usage. Composition and probing are blocking work, so they
run in a worker thread; local composition has no monetary cost, so only render time, size and CPU are recorded."""

from __future__ import annotations

import asyncio

from app.observability.scope import ExecutionScope
from app.providers.video.base import VideoCompositionError, VideoProbeError
from app.schemas.artifact import Artifact, StoredObject
from app.schemas.events import EventType
from app.schemas.video import (
    VideoArtifactRequest,
    VideoComposeRequest,
    VideoComposeResult,
    VideoValidationInput,
    VideoValidationReport,
)
from app.tools.base import Tool, ToolError
from app.tools.video.service import VideoArtifactRejected, VideoService


class VideoCompositionFailed(ToolError):
    pass


class VideoValidationFailed(ToolError):
    pass


def _task(scope: ExecutionScope) -> str:
    if scope.task_id is None:
        raise ToolError("videos can only be made inside a task")
    return scope.task_id


class VideoComposeTool(Tool[VideoComposeRequest, VideoComposeResult]):
    name = "video.compose"
    description = "Compose a validated VideoPlan into a video file with the configured composer and store it in " \
                  "the object store; reuse this task's VIDEO artifact when an equivalent one already exists."
    input_model = VideoComposeRequest
    output_model = VideoComposeResult
    permissions = frozenset({"artifact:read", "artifact:write"})
    timeout_seconds = 1800.0

    def __init__(self, service: VideoService) -> None:
        self._service = service

    async def run(self, data: VideoComposeRequest, scope: ExecutionScope) -> VideoComposeResult:
        task_id, plan = _task(scope), data.plan
        key = self._service.composition_key(plan)
        existing = self._service.find(task_id, data.name, key)
        if existing is not None:
            scope.emit(EventType.VIDEO_COMPOSITION_COMPLETED, tool=self.name, video_plan_id=plan.video_plan_id,
                       composition_key=key, reused=True, artifact_id=existing.artifact_id,
                       checksum=existing.content_hash, size_bytes=existing.size_bytes)
            return VideoComposeResult(
                composition_key=key, object=_object_of(existing), composer=existing.metadata.get("composer", ""),
                duration=float(existing.metadata.get("duration", 0.0)), reused=True, artifact=existing,
                subtitles_burned=int(existing.metadata.get("subtitles_burned", 0)))
        composer = self._service.composer.name
        scope.emit(EventType.VIDEO_COMPOSITION_STARTED, tool=self.name, video_plan_id=plan.video_plan_id,
                   composition_key=key, composer=composer, segments=len(plan.slides),
                   audio_tracks=len(plan.audio_tracks), width=plan.resolution.width, height=plan.resolution.height,
                   fps=plan.fps, duration=plan.duration)
        try:
            result = await asyncio.to_thread(self._service.compose, plan, key)
        except (VideoCompositionError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            scope.emit(EventType.VIDEO_FAILED, tool=self.name, stage="compose", video_plan_id=plan.video_plan_id,
                       composer=composer, error=error[:1000])
            raise VideoCompositionFailed(f"composing video {plan.video_plan_id} failed: {error}") from exc
        scope.emit(EventType.VIDEO_COMPOSITION_COMPLETED, tool=self.name, video_plan_id=plan.video_plan_id,
                   composition_key=key, reused=False, composer=result.composer, frames=result.frames,
                   duration=result.duration, render_seconds=result.render_seconds, cpu_seconds=result.cpu_seconds,
                   checksum=result.object.checksum, size_bytes=result.object.size_bytes,
                   reused_object=result.object.reused)
        units = {"videos": 1, "render_seconds": result.render_seconds, "output_bytes": result.object.size_bytes,
                 "frames": result.frames, "video_seconds": round(result.duration, 3)}
        if result.cpu_seconds is not None:
            units["cpu_seconds"] = result.cpu_seconds
        scope.usage.record_service(service=f"video_compose:{result.composer.split('/')[0]}", results=1, units=units)
        return result


def _object_of(artifact: Artifact) -> StoredObject:
    return StoredObject(uri=artifact.uri, checksum=artifact.content_hash, key=artifact.metadata.get("object_key"),
                        media_type=artifact.media_type, size_bytes=artifact.size_bytes, reused=True)


class VideoValidateTool(Tool[VideoValidationInput, VideoValidationReport]):
    name = "video.validate"
    description = "Validate a composed video with a real container parser against its plan: MP4 container, " \
                  "checksum, duration, resolution, frame rate, codecs, audio stream, narration, frames."
    input_model = VideoValidationInput
    output_model = VideoValidationReport
    permissions = frozenset({"artifact:read"})
    timeout_seconds = 600.0

    def __init__(self, service: VideoService) -> None:
        self._service = service

    async def run(self, data: VideoValidationInput, scope: ExecutionScope) -> VideoValidationReport:
        plan, composed = data.plan, data.composed
        scope.emit(EventType.VIDEO_VALIDATION_STARTED, tool=self.name, video_plan_id=plan.video_plan_id,
                   checksum=composed.object.checksum, expected_duration=plan.duration,
                   validator=self._service.validator.name, prober=self._service.prober.name)
        try:
            report = await asyncio.to_thread(self._service.validate, plan, composed)
        except (VideoProbeError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            scope.emit(EventType.VIDEO_FAILED, tool=self.name, stage="validate", video_plan_id=plan.video_plan_id,
                       error=error[:1000])
            raise VideoValidationFailed(f"validating video {plan.video_plan_id} failed: {error}") from exc
        probe = report.probe
        scope.emit(EventType.VIDEO_VALIDATION_COMPLETED, tool=self.name, video_plan_id=plan.video_plan_id,
                   valid=report.valid, checks=report.checks, errors=[e.model_dump(exclude_none=True)
                                                                     for e in report.errors][:20],
                   duration=probe.duration if probe else None,
                   width=probe.video.width if probe and probe.video else None,
                   height=probe.video.height if probe and probe.video else None,
                   fps=probe.video.fps if probe and probe.video else None,
                   audio_streams=len(probe.audio) if probe else 0)
        if not report.valid:
            scope.emit(EventType.VIDEO_FAILED, tool=self.name, stage="validate", video_plan_id=plan.video_plan_id,
                       error="; ".join(f"{e.code}: {e.message}" for e in report.errors)[:1000])
        return report


class VideoArtifactTool(Tool[VideoArtifactRequest, Artifact]):
    name = "video.create_artifact"
    description = "Create the VIDEO artifact for a composed video that passed validation; it points at the " \
                  "stored MP4 and links the video plan, the timeline and the presentation."
    input_model = VideoArtifactRequest
    output_model = Artifact
    permissions = frozenset({"artifact:write"})

    def __init__(self, service: VideoService) -> None:
        self._service = service

    async def run(self, data: VideoArtifactRequest, scope: ExecutionScope) -> Artifact:
        task_id = _task(scope)
        previous = self._service.artifacts.find(task_id, data.name)
        try:
            artifact = self._service.store(task_id, data, scope)
        except VideoArtifactRejected as exc:
            scope.emit(EventType.VIDEO_FAILED, tool=self.name, stage="artifact",
                       video_plan_id=data.plan.video_plan_id, error=str(exc))
            raise ToolError(str(exc)) from exc
        scope.emit(EventType.VIDEO_ARTIFACT_CREATED, tool=self.name, artifact_id=artifact.artifact_id,
                   version=artifact.version, reused=previous is not None and previous.artifact_id == artifact.artifact_id,
                   checksum=artifact.content_hash, size_bytes=artifact.size_bytes, media_type=artifact.media_type,
                   duration=artifact.metadata["duration"], parent_ids=artifact.parent_ids)
        return artifact
