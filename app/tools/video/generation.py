"""Generated video clips: the workflow's only way to submit, follow, cancel and store a generated clip.

    GenerativeVideoNode (runtime: owns the waiting) -> generation tools -> GeneratedVideoService
        -> VideoGenerationProvider (interface; mock or a vendor adapter, chosen in the composition root)
        -> ClipValidator + VideoNormalizer (FFmpeg, or the mock normaliser in tests)

Idempotency: a generation key identifies what is asked (the provider request without our metadata, the provider
and its model). The generation ledger, a small record per key in the object store, remembers the job and, once
downloaded, the provider's file (content-addressed) and its normalised clips. So a resumed task polls its existing
job instead of submitting again, and any task asking for the same clip later reuses the stored file instead of
paying for it twice. Records hold ids and object references only, never prompts' owners or learner data.

Nothing about a clip is trusted until measured: the provider's file is validated from its bytes, normalised to the
platform format (VideoComposer never sees a provider format), then validated again before a GENERATED_VIDEO_ASSET
artifact points at it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.providers.core.errors import ProviderError
from app.providers.video.base import VideoNormalizationError, VideoNormalizer, VideoProbeError, VideoProber
from app.providers.video_generation.base import VideoGenerationProvider
from app.schemas.artifact import ArtifactType, StoredObject
from app.schemas.common import RetryPolicy
from app.schemas.events import EventType
from app.schemas.generative_video import (
    ClipAssetRequest,
    ClipAssetResult,
    ClipExpectation,
    ClipValidationReport,
    GeneratedVideoConfig,
    GenerationJobRequest,
    GenerationLookup,
    GenerationLookupRequest,
    GenerationSubmitRequest,
    ProviderVideoLimits,
    VideoGenerationJob,
    VideoGenerationStatus,
    VideoSegmentAsset,
    VideoSegmentPlan,
)
from app.schemas.video import ClipNormalization
from app.tools.base import Tool, ToolError, ToolTransientError
from app.tools.video.clips import ClipValidator
from app.utils.workspace import ScratchSpace

LEDGER = "video-generations"
SOURCE_SUFFIX = {"video/mp4": ".mp4", "video/x-msvideo": ".avi", "video/webm": ".webm", "video/quicktime": ".mov"}


class ClipRejected(ToolError):
    """A clip could not be turned into a GENERATED_VIDEO_ASSET. `stage` says where (download, validate, normalize,
    artifact); `report` is the failed validation, when there was one."""

    def __init__(self, message: str, *, stage: str, report: ClipValidationReport | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.report = report


def asset_name(segment_id: str) -> str:
    return f"generated_video_{segment_id}"


def _errors(report: ClipValidationReport) -> str:
    return "; ".join(f"{e.code} ({e.field}): {e.message}" + (f" expected {e.expected}, got {e.actual}"
                                                              if e.expected else "") for e in report.errors)


def _task(scope: ExecutionScope) -> str:
    if scope.task_id is None:
        raise ToolError("generated clips can only be made inside a task")
    return scope.task_id


class GeneratedVideoService:
    def __init__(self, artifacts: ArtifactService, provider: VideoGenerationProvider, prober: VideoProber,
                 normalizer: VideoNormalizer, scratch: ScratchSpace,
                 config: GeneratedVideoConfig | None = None) -> None:
        self.artifacts = artifacts
        self.provider = provider
        self.prober = prober
        self.normalizer = normalizer
        self.scratch = scratch
        self.config = config or GeneratedVideoConfig()
        self.validator = ClipValidator(prober)

    def limits(self) -> ProviderVideoLimits:
        return self.provider.limits()

    def generation_key(self, segment: VideoSegmentPlan) -> str:
        body = json.dumps({"request": segment.request().request_hash(), "provider": self.provider.name,
                           "model": self.provider.model}, sort_keys=True)
        return "vgk_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]

    # --- the ledger ---------------------------------------------------------------------------------------------

    def _record(self, key: str) -> dict:
        return self.artifacts.read_record(LEDGER, key) or {}

    def lookup(self, key: str) -> GenerationLookup:
        record = self._record(key)
        job = VideoGenerationJob.model_validate(record["job"]) if record.get("job") else None
        raw = StoredObject.model_validate(record["raw"]) if record.get("raw") else None
        if raw is not None and not self.artifacts.object_exists(raw.uri):
            raw = None
        return GenerationLookup(generation_key=key, job=job, raw=raw)

    def remember(self, job: VideoGenerationJob, **fields: object) -> None:
        record = self._record(job.generation_key)
        record.update({"job": job.model_dump(mode="json"), **fields})
        self.artifacts.put_record(LEDGER, job.generation_key, record)

    # --- provider steps -----------------------------------------------------------------------------------------

    async def submit(self, segment: VideoSegmentPlan, key: str) -> VideoGenerationJob:
        request = segment.request()
        self.provider.check(request)
        job = await self.provider.submit(request, generation_key=key)
        self.remember(job)
        return job

    async def refresh(self, job: VideoGenerationJob) -> VideoGenerationJob:
        updated = await self.provider.status(job)
        updated = updated.model_copy(update={"polls": job.polls + 1})
        self.remember(updated)
        return updated

    async def cancel(self, job: VideoGenerationJob) -> VideoGenerationJob:
        cancelled = await self.provider.cancel(job)
        self.remember(cancelled)
        return cancelled

    async def source(self, job: VideoGenerationJob) -> tuple[StoredObject, bool, dict]:
        """The provider's file for a completed job: from the ledger when it was downloaded before (reused=True),
        otherwise downloaded now and stored content-addressed. Returns (object, reused, provider claims)."""
        known = self.lookup(job.generation_key).raw
        if known is not None:
            return known, True, {}
        file = await self.provider.download(job)
        if not file.content:
            raise ClipRejected(f"provider '{job.provider}' returned an empty file for job {job.provider_job_id}",
                               stage="download")
        raw = self.artifacts.put_object(file.content, file.media_type)
        self.remember(job, raw=raw.model_dump(mode="json"))
        claims = {"reported_duration": file.reported_duration, "reported_width": file.reported_width,
                  "reported_height": file.reported_height, "reported_fps": file.reported_fps,
                  "model": file.model, "provider_cost_usd": file.usage.cost_usd}
        return raw, False, claims

    # --- validation, normalisation and the artifact -------------------------------------------------------------

    def target(self, segment: VideoSegmentPlan) -> ClipNormalization:
        return ClipNormalization(width=segment.width, height=segment.height, fps=segment.fps,
                                 duration=segment.duration, keep_audio=segment.audio == "mixed")

    def raw_expectation(self, segment: VideoSegmentPlan) -> ClipExpectation:
        return ClipExpectation(stage="raw", duration=segment.duration,
                               duration_tolerance=self.config.duration_tolerance, aspect_ratio=segment.aspect_ratio,
                               min_width=self.config.min_width, min_height=self.config.min_height,
                               audio="required" if segment.audio == "mixed" else "any")

    def normalized_expectation(self, segment: VideoSegmentPlan, target: ClipNormalization) -> ClipExpectation:
        return ClipExpectation(stage="normalized", duration=segment.duration,
                               duration_tolerance=max(self.config.duration_tolerance, 1.0 / target.fps),
                               aspect_ratio=segment.aspect_ratio, width=target.width, height=target.height,
                               fps=float(target.fps), codecs=[target.codec.value],
                               containers=[target.container.value],
                               audio="required" if target.keep_audio else "absent")

    def prepare(self, segment: VideoSegmentPlan, job: VideoGenerationJob,
                raw: StoredObject) -> tuple[StoredObject, ClipValidationReport, ClipValidationReport, str]:
        """Validate the provider's file, normalise it (or reuse an earlier normalisation of the same file to the same
        target) and validate the result. Returns (clip, raw report, clip report, normaliser)."""
        target = self.target(segment)
        target_key = hashlib.sha256(f"{self.normalizer.name}:{target.model_dump_json()}".encode()).hexdigest()[:16]
        with self.scratch.session(f"clip-{segment.segment_id}") as workspace:
            source = workspace / ("source" + SOURCE_SUFFIX.get(raw.media_type, ".bin"))
            checksum = self.artifacts.copy_object_to(raw.uri, source)
            raw_report = self.validator.validate(source, raw.checksum, self.raw_expectation(segment))
            if checksum != raw.checksum and raw_report.valid:  # copy_object_to measured other bytes than stored
                raise ClipRejected("the stored provider file changed after it was stored", stage="validate",
                                   report=raw_report)
            if not raw_report.valid:
                raise ClipRejected(f"the provider's clip for {segment.segment_id} is not usable: "
                                   f"{_errors(raw_report)}", stage="validate", report=raw_report)
            normalized = self._reusable_clip(job.generation_key, target_key)
            if normalized is None:
                probe = self.prober.probe(source)
                try:
                    clip = self.normalizer.normalize(source, probe, target, workspace)
                except (VideoNormalizationError, VideoProbeError, OSError, ValueError) as exc:
                    raise ClipRejected(f"normalising the clip for {segment.segment_id} failed: {exc}",
                                       stage="normalize") from exc
                normalized = self.artifacts.put_object_file(Path(clip.path), clip.media_type)
            local = workspace / "clip.mp4"
            self.artifacts.copy_object_to(normalized.uri, local)
            report = self.validator.validate(local, normalized.checksum,
                                             self.normalized_expectation(segment, target))
            if not report.valid:
                raise ClipRejected(f"the normalised clip for {segment.segment_id} is not valid: {_errors(report)}",
                                   stage="normalize", report=report)
        record = self._record(job.generation_key)
        record.setdefault("normalized", {})[target_key] = normalized.model_dump(mode="json")
        self.artifacts.put_record(LEDGER, job.generation_key, record)
        return normalized, raw_report, report, self.normalizer.name

    def _reusable_clip(self, key: str, target_key: str) -> StoredObject | None:
        found = self._record(key).get("normalized", {}).get(target_key)
        if not found:
            return None
        obj = StoredObject.model_validate(found)
        return obj if self.artifacts.object_exists(obj.uri) else None

    def store(self, task_id: str, data: ClipAssetRequest, raw: StoredObject, reused: bool, claims: dict,
              scope: ExecutionScope) -> ClipAssetResult:
        segment, job = data.segment, data.job
        clip, raw_report, report, normalizer = self.prepare(segment, job, raw)
        asset_id = "gva_" + hashlib.sha256(f"{segment.segment_id}:{clip.checksum}".encode()).hexdigest()[:16]
        metadata = {
            "asset_id": asset_id, "segment_id": segment.segment_id, "lesson_section_id": segment.lesson_section_id,
            "slide_id": segment.slide_id, "plan_id": data.plan_id, "purpose": segment.purpose.value,
            "insertion_strategy": segment.insertion_strategy.value, "audio": segment.audio,
            "duration": report.duration, "width": report.width, "height": report.height, "fps": report.fps,
            "format": report.container, "codec": report.codec, "has_audio": report.has_audio,
            "object_key": clip.key or "", "checksum": clip.checksum, "provider": job.provider, "model": job.model,
            "generation_job_id": job.job_id, "provider_job_id": job.provider_job_id,
            "generation_key": job.generation_key, "source_checksum": raw.checksum,
            "source_media_type": raw.media_type, "normalizer": normalizer, "reused": reused,
            "provider_claims": {k: v for k, v in claims.items() if v is not None},
            "raw_validation": raw_report.model_dump(mode="json"), "validation": report.model_dump(mode="json"),
        }
        try:
            artifact = self.artifacts.store_object(
                task_id=task_id, name=asset_name(segment.segment_id), type=ArtifactType.GENERATED_VIDEO_ASSET,
                obj=clip, provider=job.provider, parent_ids=data.parent_ids, metadata=metadata, scope=scope)
        except (KeyError, ValueError) as exc:
            raise ClipRejected(f"storing the clip for {segment.segment_id} failed: {exc}", stage="artifact") from exc
        asset = VideoSegmentAsset(
            asset_id=asset_id, segment_id=segment.segment_id, artifact_id=artifact.artifact_id, uri=artifact.uri,
            object_key=clip.key or "", checksum=clip.checksum, media_type=clip.media_type,
            duration=report.duration or segment.duration, width=report.width or segment.width,
            height=report.height or segment.height, fps=report.fps or segment.fps, format=report.container or "mp4",
            has_audio=report.has_audio, provider=job.provider, model=job.model, generation_job_id=job.job_id,
            provider_job_id=job.provider_job_id, generation_key=job.generation_key, source_checksum=raw.checksum,
            reused=reused, metadata={"normalizer": normalizer, "plan_id": data.plan_id})
        return ClipAssetResult(asset=asset, artifact=artifact, raw_validation=raw_report, validation=report)


def _provider_failure(service: GeneratedVideoService, what: str, exc: Exception) -> ToolError:
    transient = getattr(exc, "transient", True)
    return (ToolTransientError if transient else ToolError)(
        f"video generation provider '{service.provider.name}' {what} failed: {exc}")


class VideoGenerationSubmitTool(Tool[GenerationSubmitRequest, VideoGenerationJob]):
    name = "video_generation.submit"
    description = ("Submit one planned clip to the configured video generation provider, or return the job already "
                   "known for the same generation key (a resumed or repeated request is never submitted twice).")
    input_model = GenerationSubmitRequest
    output_model = VideoGenerationJob
    permissions = frozenset({"media:generate"})
    timeout_seconds = 120.0
    retry = RetryPolicy(max_attempts=1)  # the provider invoker already retries; a submit is billed

    def __init__(self, service: GeneratedVideoService) -> None:
        self._service = service

    async def run(self, data: GenerationSubmitRequest, scope: ExecutionScope) -> VideoGenerationJob:
        segment = data.segment
        key = data.generation_key or self._service.generation_key(segment)
        known = self._service.lookup(key)
        if known.job is not None and known.job.status in (VideoGenerationStatus.SUBMITTED,
                                                          VideoGenerationStatus.PROCESSING,
                                                          VideoGenerationStatus.COMPLETED):
            scope.emit(EventType.VIDEO_GENERATION_REUSED, tool=self.name, segment_id=segment.segment_id,
                       generation_key=key, job_id=known.job.job_id, provider=known.job.provider,
                       status=known.job.status.value, downloaded=known.raw is not None)
            return known.job
        try:
            job = await self._service.submit(segment, key)
        except (ProviderError, ConnectionError, OSError) as exc:
            scope.emit(EventType.VIDEO_GENERATION_FAILED, tool=self.name, stage="submit",
                       segment_id=segment.segment_id, generation_key=key, error=str(exc)[:1000])
            raise _provider_failure(self._service, "submission", exc) from exc
        scope.emit(EventType.VIDEO_GENERATION_SUBMITTED, tool=self.name, segment_id=segment.segment_id,
                   generation_key=key, job_id=job.job_id, provider=job.provider, model=job.model,
                   provider_job_id=job.provider_job_id, duration=segment.duration,
                   estimated_cost_usd=segment.estimated_cost_usd)
        return job


class VideoGenerationStatusTool(Tool[GenerationJobRequest, VideoGenerationJob]):
    name = "video_generation.status"
    description = "Ask the provider that owns a video generation job for its current status (one poll)."
    input_model = GenerationJobRequest
    output_model = VideoGenerationJob
    timeout_seconds = 60.0
    retry = RetryPolicy(max_attempts=1)  # the workflow's poller decides when to ask again

    def __init__(self, service: GeneratedVideoService) -> None:
        self._service = service

    async def run(self, data: GenerationJobRequest, scope: ExecutionScope) -> VideoGenerationJob:
        job = data.job
        try:
            updated = await self._service.refresh(job)
        except (ProviderError, ConnectionError, OSError) as exc:
            raise _provider_failure(self._service, f"status of job {job.job_id}", exc) from exc
        scope.emit(EventType.VIDEO_GENERATION_POLLED, tool=self.name, job_id=job.job_id,
                   segment_id=job.request.metadata.get("segment_id"), status=updated.status.value,
                   polls=updated.polls)
        if updated.status == VideoGenerationStatus.COMPLETED:
            scope.emit(EventType.VIDEO_GENERATION_COMPLETED, tool=self.name, job_id=job.job_id,
                       provider=updated.provider, provider_job_id=updated.provider_job_id, polls=updated.polls)
        elif updated.status == VideoGenerationStatus.FAILED:
            scope.emit(EventType.VIDEO_GENERATION_FAILED, tool=self.name, stage="poll", job_id=job.job_id,
                       provider=updated.provider, error=(updated.error or "")[:1000])
        return updated


class VideoGenerationCancelTool(Tool[GenerationJobRequest, VideoGenerationJob]):
    name = "video_generation.cancel"
    description = ("Cancel a video generation job: at the provider when it supports cancellation, otherwise "
                   "recorded as a local cancellation (its result is never used).")
    input_model = GenerationJobRequest
    output_model = VideoGenerationJob
    timeout_seconds = 60.0

    def __init__(self, service: GeneratedVideoService) -> None:
        self._service = service

    async def run(self, data: GenerationJobRequest, scope: ExecutionScope) -> VideoGenerationJob:
        job = data.job
        try:
            cancelled = await self._service.cancel(job)
        except (ProviderError, ConnectionError, OSError) as exc:
            # the provider could not be reached: the job is still never used, so record a local cancellation
            cancelled = job.advanced(VideoGenerationStatus.CANCELLED, error=f"provider cancel failed: {exc}")
            cancelled = cancelled.model_copy(update={"cancellation": "local"})
            self._service.remember(cancelled)
        scope.emit(EventType.VIDEO_GENERATION_CANCELLED, tool=self.name, job_id=job.job_id,
                   provider=cancelled.provider, cancellation=cancelled.cancellation,
                   segment_id=job.request.metadata.get("segment_id"))
        return cancelled


class VideoGenerationLookupTool(Tool[GenerationLookupRequest, GenerationLookup]):
    name = "video_generation.lookup"
    description = "What the generation ledger knows about a generation key: its job and whether its file is stored."
    input_model = GenerationLookupRequest
    output_model = GenerationLookup

    def __init__(self, service: GeneratedVideoService) -> None:
        self._service = service

    async def run(self, data: GenerationLookupRequest, scope: ExecutionScope) -> GenerationLookup:
        return self._service.lookup(data.generation_key)


class GeneratedClipAssetTool(Tool[ClipAssetRequest, ClipAssetResult]):
    name = "video_generation.create_asset"
    description = ("Download (or reuse) a completed job's clip, validate it from its bytes, normalise it to the "
                   "platform format, validate it again and store it as a GENERATED_VIDEO_ASSET artifact.")
    input_model = ClipAssetRequest
    output_model = ClipAssetResult
    permissions = frozenset({"artifact:read", "artifact:write"})
    timeout_seconds = 900.0
    retry = RetryPolicy(max_attempts=1)

    def __init__(self, service: GeneratedVideoService) -> None:
        self._service = service

    async def run(self, data: ClipAssetRequest, scope: ExecutionScope) -> ClipAssetResult:
        task_id, segment, job = _task(scope), data.segment, data.job
        if job.status != VideoGenerationStatus.COMPLETED:
            raise ClipRejected(f"job {job.job_id} is {job.status.value}, not completed", stage="download")
        try:
            raw, reused, claims = await self._service.source(job)
        except ClipRejected:
            raise
        except (ProviderError, ConnectionError, OSError) as exc:
            scope.emit(EventType.VIDEO_GENERATION_FAILED, tool=self.name, stage="download", job_id=job.job_id,
                       error=str(exc)[:1000])
            raise ClipRejected(f"downloading job {job.job_id} failed: {exc}", stage="download") from exc
        if reused:
            scope.emit(EventType.VIDEO_GENERATION_REUSED, tool=self.name, segment_id=segment.segment_id,
                       generation_key=job.generation_key, job_id=job.job_id, checksum=raw.checksum, stored=True)
        try:
            result = await asyncio.to_thread(self._service.store, task_id, data, raw, reused, claims, scope)
        except ClipRejected as exc:
            if exc.report is not None:
                scope.emit(EventType.GENERATED_VIDEO_VALIDATED, tool=self.name, segment_id=segment.segment_id,
                           stage=exc.report.stage, valid=False, checksum=exc.report.checksum,
                           errors=[e.model_dump(exclude_none=True) for e in exc.report.errors][:20])
            raise
        for report in (result.raw_validation, result.validation):
            scope.emit(EventType.GENERATED_VIDEO_VALIDATED, tool=self.name, segment_id=segment.segment_id,
                       stage=report.stage, valid=report.valid, checksum=report.checksum, checks=report.checks,
                       container=report.container, codec=report.codec, duration=report.duration,
                       width=report.width, height=report.height, fps=report.fps, has_audio=report.has_audio)
        asset, artifact = result.asset, result.artifact
        scope.emit(EventType.GENERATED_VIDEO_ASSET_CREATED, tool=self.name, segment_id=segment.segment_id,
                   artifact_id=artifact.artifact_id, asset_id=asset.asset_id, checksum=asset.checksum,
                   duration=asset.duration, width=asset.width, height=asset.height, fps=asset.fps,
                   provider=asset.provider, generation_job_id=asset.generation_job_id, reused=asset.reused,
                   parent_ids=artifact.parent_ids)
        return result


def generation_tools(service: GeneratedVideoService) -> list[Tool]:
    return [VideoGenerationSubmitTool(service), VideoGenerationStatusTool(service), VideoGenerationCancelTool(service),
            VideoGenerationLookupTool(service), GeneratedClipAssetTool(service)]

