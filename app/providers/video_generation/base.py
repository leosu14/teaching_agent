"""Video generation provider contract. Adapters (MiniMax, ...) map their API onto these schemas; nothing here is
vendor-specific.

Generation is asynchronous: `submit` returns a job, `status` reports its progress, `download` returns the finished
file and `cancel` stops it (at the provider when it supports cancellation, otherwise the cancellation is recorded as
local). The workflow drives these steps itself and owns the waiting; `generate` runs all of them in one call, with
the same bounded generic poller, for scripts and smoke tests.

A provider never decides educational intent: it receives a finished prompt and technical parameters only.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import ClassVar

from app.providers.core.base import Provider
from app.providers.core.errors import ProviderError, ProviderInvalidRequest, ProviderResponseError, is_retryable
from app.schemas.generative_video import (
    GeneratedVideoFile,
    ProviderVideoLimits,
    VideoGenerationJob,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoGenerationStatus,
)
from app.schemas.providers import Capability
from app.utils.polling import PollPolicy, poll_until


def job_id_for(generation_key: str) -> str:
    """Our job id: deterministic for a generation key, so a resumed task finds its job instead of submitting again."""
    return "vgj_" + hashlib.sha256(generation_key.encode("utf-8")).hexdigest()[:20]


class VideoGenerationProvider(Provider, ABC):
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.VIDEO_GENERATION})
    supports_seed: bool = False
    supports_cancel: bool = False
    supports_audio: bool = False
    durations: tuple[float, ...] | None = None  # the only durations offered; None: any up to max_duration
    max_duration: float = 10.0
    aspect_ratios: tuple[str, ...] = ("16:9",)
    model: str | None = None

    def limits(self) -> ProviderVideoLimits:
        return ProviderVideoLimits(
            provider=self.name, model=self.model, durations=list(self.durations) if self.durations else None,
            max_duration=self.max_duration, aspect_ratios=list(self.aspect_ratios), supports_audio=self.supports_audio,
            supports_seed=self.supports_seed, supports_cancel=self.supports_cancel,
            requires_network=self.requires_network)

    def check(self, request: VideoGenerationRequest) -> None:
        """Reject a request this provider cannot serve before anything is sent. Arbitrary durations are never
        rounded silently: the strategy only plans durations the provider offers."""
        if request.aspect_ratio not in self.aspect_ratios:
            raise ProviderInvalidRequest(f"{self.name}: aspect ratio {request.aspect_ratio} is not supported "
                                         f"(supported: {', '.join(self.aspect_ratios)})", provider=self.name)
        if self.durations is not None and not any(abs(request.duration - d) < 1e-6 for d in self.durations):
            raise ProviderInvalidRequest(f"{self.name}: a {request.duration:g}s clip is not offered (durations: "
                                         f"{', '.join(f'{d:g}s' for d in self.durations)})", provider=self.name)
        if request.duration > self.max_duration + 1e-6:
            raise ProviderInvalidRequest(f"{self.name}: {request.duration:g}s is longer than the "
                                         f"{self.max_duration:g}s maximum", provider=self.name)
        if request.with_audio and not self.supports_audio:
            raise ProviderInvalidRequest(f"{self.name}: generated audio is not supported", provider=self.name)

    @abstractmethod
    async def submit(self, request: VideoGenerationRequest, *, generation_key: str) -> VideoGenerationJob: ...

    @abstractmethod
    async def status(self, job: VideoGenerationJob) -> VideoGenerationJob: ...

    @abstractmethod
    async def download(self, job: VideoGenerationJob) -> GeneratedVideoFile: ...

    async def cancel(self, job: VideoGenerationJob) -> VideoGenerationJob:
        """Stop a job. Providers without cancellation record it as local: the job may still run (and bill) at the
        vendor, but nothing here will use its result."""
        return job.advanced(VideoGenerationStatus.CANCELLED).model_copy(update={"cancellation": "local"})

    async def generate(self, request: VideoGenerationRequest, *, poll: PollPolicy | None = None,
                       generation_key: str | None = None) -> VideoGenerationResult:
        """Submit, wait (bounded) and download in one call. Raises ProviderError when the job does not complete."""
        key = generation_key or f"{self.name}:{self.model}:{request.request_hash()}"
        job = await self.submit(request, generation_key=key)
        outcome = await poll_until(lambda _n: self.status(job), lambda j: j.status.terminal,
                                   poll or PollPolicy(), retryable=lambda e: isinstance(e, ProviderError)
                                   and is_retryable(e))
        final = outcome.value or job
        if final.status != VideoGenerationStatus.COMPLETED:
            reason = final.error or (f"still {final.status.value} after {outcome.attempts} polls"
                                     if not final.status.terminal else final.status.value)
            raise ProviderResponseError(f"{self.name}: video job {final.provider_job_id} did not complete: {reason}",
                                        provider=self.name, transient=not final.status.terminal)
        file = await self.download(final)
        return VideoGenerationResult(
            provider=self.name, provider_job_id=final.provider_job_id, status=final.status, content=file.content,
            duration=file.reported_duration, width=file.reported_width, height=file.reported_height,
            fps=file.reported_fps, media_type=file.media_type, model=file.model, usage=file.usage,
            metadata={**file.metadata, "job_id": final.job_id})
