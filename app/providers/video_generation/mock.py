"""MockVideoGenerationProvider: deterministic fixture clips, offline. The default video generation provider.

Every job renders a small, genuinely playable Motion-JPEG AVI (a moving shape over colour bands derived from the
request hash), so the whole pipeline (validation from bytes, normalisation, composition) runs on real video without
a network call or a paid API. Same request, same bytes. The clip is small on purpose (192 px wide, 12 fps): the
normalisation stage brings it to the platform format like any provider's output.

Jobs are stateless at the "vendor": a job id encodes nothing secret, and a fresh instance (a restarted process) can
poll and download a job another instance submitted. Faults can be injected for tests.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

from PIL import Image, ImageDraw

from app.providers.core.errors import ProviderInvalidRequest, ProviderUnavailable
from app.providers.video.avi import jpeg_frame, write_mjpeg_avi
from app.providers.video_generation.base import VideoGenerationProvider, job_id_for
from app.schemas.generative_video import (
    GeneratedVideoFile,
    VideoGenerationJob,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoGenerationUsage,
)

MOCK_WIDTH = 192
MOCK_FPS = 12


def render_clip(request: VideoGenerationRequest, *, width: int = MOCK_WIDTH, fps: int = MOCK_FPS) -> bytes:
    """The fixture clip for a request: `request.duration` seconds of a moving circle over three colour bands."""
    num, den = (int(x) for x in request.aspect_ratio.split(":"))
    height = max(2, round(width * den / num / 2) * 2)
    digest = hashlib.sha256(request.request_hash().encode("utf-8")).digest()
    bands = [tuple(digest[3 * i:3 * i + 3]) for i in range(3)]
    accent = tuple(255 - c for c in digest[9:12])
    frames = []
    total = max(1, round(request.duration * fps))
    radius = max(4, height // 5)
    for n in range(total):
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)
        for i, color in enumerate(bands):
            draw.rectangle((0, i * height // 3, width, (i + 1) * height // 3), fill=color)
        x = round((width + 2 * radius) * n / total) - radius
        y = height // 2 + round((height // 4) * ((n % fps) / fps - 0.5))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=accent, outline=(255, 255, 255))
        draw.rectangle((0, height - 4, round(width * (n + 1) / total), height), fill=(255, 255, 255))
        frames.append(jpeg_frame(img))
    return write_mjpeg_avi(frames, width, height, fps)


class MockVideoGenerationProvider(VideoGenerationProvider):
    name = "mock"
    model = "mock-video-1"
    requires_network: ClassVar[bool] = False
    supports_seed = True
    supports_cancel = True
    supports_audio = False
    durations = None
    max_duration = 10.0
    aspect_ratios = ("16:9", "4:3", "1:1", "9:16")

    def __init__(self, *, processing_polls: int = 1, fail_submit: bool = False, fail_job: bool = False,
                 status_errors: int = 0, corrupt: bool = False, duration_offset: float = 0.0) -> None:
        self.processing_polls = processing_polls  # PROCESSING answers before a job completes
        self.fail_submit = fail_submit
        self.fail_job = fail_job
        self.status_errors = status_errors  # transient errors raised by the next status calls
        self.corrupt = corrupt  # download returns a truncated file
        self.duration_offset = duration_offset  # renders a clip this many seconds off what was asked
        self.requests: list[VideoGenerationRequest] = []
        self.submits = 0
        self.status_calls = 0
        self.downloads = 0
        self.cancels = 0
        self._polls: dict[str, int] = {}

    def configuration(self) -> dict:
        return {"model": self.model, "fixture": f"{MOCK_WIDTH}px MJPEG AVI at {MOCK_FPS} fps"}

    async def submit(self, request: VideoGenerationRequest, *, generation_key: str) -> VideoGenerationJob:
        self.check(request)
        self.requests.append(request)
        if self.fail_submit:
            raise ProviderInvalidRequest("mock video provider configured to reject submissions", provider=self.name)
        self.submits += 1
        provider_job_id = "mockjob_" + hashlib.sha256(generation_key.encode("utf-8")).hexdigest()[:16]
        return VideoGenerationJob(job_id=job_id_for(generation_key), provider=self.name,
                                  provider_job_id=provider_job_id, status=VideoGenerationStatus.SUBMITTED,
                                  request=request, generation_key=generation_key, model=self.model)

    async def status(self, job: VideoGenerationJob) -> VideoGenerationJob:
        self.status_calls += 1
        if self.status_errors > 0:
            self.status_errors -= 1
            raise ProviderUnavailable("mock video provider: temporarily unavailable", provider=self.name)
        if job.status.terminal:
            return job
        if self.fail_job:
            return job.advanced(VideoGenerationStatus.FAILED, error="mock video provider configured to fail jobs")
        seen = self._polls.get(job.provider_job_id, 0) + 1
        self._polls[job.provider_job_id] = seen
        status = VideoGenerationStatus.PROCESSING if seen <= self.processing_polls else VideoGenerationStatus.COMPLETED
        return job.advanced(status).model_copy(update={"polls": job.polls + 1})

    async def download(self, job: VideoGenerationJob) -> GeneratedVideoFile:
        if job.status != VideoGenerationStatus.COMPLETED:
            raise ProviderInvalidRequest(f"mock video job {job.provider_job_id} is {job.status.value}",
                                         provider=self.name)
        self.downloads += 1
        request = job.request
        if self.duration_offset:
            request = request.model_copy(update={"duration": max(0.5, request.duration + self.duration_offset)})
        content = render_clip(request)
        if self.corrupt:
            content = content[: len(content) * 2 // 3]
        return GeneratedVideoFile(
            content=content, media_type="video/x-msvideo", model=self.model,
            reported_duration=job.request.duration, reported_width=MOCK_WIDTH, reported_fps=float(MOCK_FPS),
            usage=VideoGenerationUsage(requests=1, seconds=job.request.duration, cost_usd=0.0),
            metadata={"fixture": "mjpeg-avi"})

    async def cancel(self, job: VideoGenerationJob) -> VideoGenerationJob:
        self.cancels += 1
        return job.advanced(VideoGenerationStatus.CANCELLED).model_copy(update={"cancellation": "provider"})
