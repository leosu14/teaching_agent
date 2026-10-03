"""Video generation adapter for the MiniMax video API (Hailuo models). Plain HTTP through HttpClient: no vendor SDK.

    POST /video_generation                 -> {"task_id", "base_resp"}               submit
    GET  /query/video_generation?task_id=  -> {"status", "file_id", "video_width", ...} status
    GET  /files/retrieve?file_id=          -> {"file": {"download_url", ...}}         download (then the URL itself)

MiniMax answers HTTP 200 with an error code in `base_resp`; those codes are mapped onto the typed provider errors.
The API offers fixed durations and resolutions: a request for anything else is rejected before it is sent (the
strategy only plans what `durations` lists). The vendor's prompt optimiser is off: the prompt comes from our
controlled prompt builder and is sent as it is. MiniMax has no job cancellation endpoint, so cancellation is local.
Nothing about the learner or the task is sent: only the prompt and the technical parameters.
"""

from __future__ import annotations

from typing import ClassVar

from app.providers.core.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderInvalidRequest,
    ProviderRateLimit,
    ProviderResponseError,
    ProviderUnavailable,
)
from app.providers.core.http import HttpClient, HttpResponse
from app.providers.video_generation.base import VideoGenerationProvider, job_id_for
from app.schemas.generative_video import (
    GeneratedVideoFile,
    VideoGenerationJob,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoGenerationUsage,
)

DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-Hailuo-02"
DURATIONS = (6.0, 10.0)
# Output sizes per resolution label. 10-second clips are only offered up to 768P.
RESOLUTIONS = {"768P": (1366, 768), "1080P": (1920, 1080)}
STATUS = {"preparing": VideoGenerationStatus.SUBMITTED, "queueing": VideoGenerationStatus.SUBMITTED,
          "processing": VideoGenerationStatus.PROCESSING, "success": VideoGenerationStatus.COMPLETED,
          "fail": VideoGenerationStatus.FAILED}
RATE_LIMIT_CODES = {1002, 1039}
AUTH_CODES = {1004, 2049}
INVALID_CODES = {1008, 2013, 2048}  # insufficient balance, invalid parameters
REFUSED_CODES = {1026, 1027}  # the prompt or the output was refused by content moderation


class MiniMaxVideoGenerationProvider(VideoGenerationProvider):
    name = "minimax"
    requires_network: ClassVar[bool] = True
    supports_seed = False
    supports_cancel = False
    supports_audio = False
    durations = DURATIONS
    max_duration = 10.0
    aspect_ratios = ("16:9",)

    def __init__(self, http: HttpClient, *, model: str | None = None, max_download_bytes: int = 200_000_000) -> None:
        self._http = http
        self.model = model or DEFAULT_MODEL
        self.max_download_bytes = max_download_bytes

    def configuration(self) -> dict:
        return {"base_url": self._http.base_url, "model": self.model, "durations": list(self.durations),
                "resolutions": sorted(RESOLUTIONS), "cancellation": "local"}

    async def probe(self) -> str:
        await self._call("GET", "/files/list", params={"purpose": "video_generation"})  # authenticated, no cost
        return "network"

    @staticmethod
    def resolution_for(request: VideoGenerationRequest) -> str:
        """The vendor resolution label: 1080P for a full-HD request of the short duration, else 768P."""
        if request.height >= 1080 and request.duration <= 6:
            return "1080P"
        return "768P"

    def request_body(self, request: VideoGenerationRequest) -> dict:
        """The JSON sent to the vendor: model, prompt and technical parameters, nothing else."""
        self.check(request)
        return {"model": self.model, "prompt": request.prompt, "duration": int(request.duration),
                "resolution": self.resolution_for(request), "prompt_optimizer": False}

    async def submit(self, request: VideoGenerationRequest, *, generation_key: str) -> VideoGenerationJob:
        body = self.request_body(request)
        data = await self._call("POST", "/video_generation", json_body=body)
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderResponseError(f"{self.name}: the submission returned no task id", provider=self.name)
        return VideoGenerationJob(job_id=job_id_for(generation_key), provider=self.name, provider_job_id=task_id,
                                  status=VideoGenerationStatus.SUBMITTED, request=request,
                                  generation_key=generation_key, model=self.model,
                                  metadata={"resolution": body["resolution"]})

    async def status(self, job: VideoGenerationJob) -> VideoGenerationJob:
        if job.status.terminal:
            return job
        data = await self._call("GET", "/query/video_generation", params={"task_id": job.provider_job_id})
        return self.parse_status(job, data)

    def parse_status(self, job: VideoGenerationJob, data: dict) -> VideoGenerationJob:
        raw = str(data.get("status", "")).strip()
        status = STATUS.get(raw.lower())
        if status is None:
            raise ProviderResponseError(f"{self.name}: unknown job status {raw!r}", provider=self.name)
        reported = {k: data[k] for k in ("file_id", "video_width", "video_height") if data.get(k) not in (None, "")}
        updated = job.advanced(status, error=f"the vendor reported the job as failed ({raw})"
                               if status == VideoGenerationStatus.FAILED else None, vendor_status=raw, **reported)
        if status == VideoGenerationStatus.COMPLETED and not reported.get("file_id"):
            raise ProviderResponseError(f"{self.name}: job {job.provider_job_id} succeeded without a file id",
                                        provider=self.name)
        return updated.model_copy(update={"polls": job.polls + 1})

    async def download(self, job: VideoGenerationJob) -> GeneratedVideoFile:
        file_id = job.metadata.get("file_id")
        if job.status != VideoGenerationStatus.COMPLETED or not file_id:
            raise ProviderInvalidRequest(f"{self.name}: job {job.provider_job_id} has no finished file",
                                         provider=self.name)
        data = await self._call("GET", "/files/retrieve", params={"file_id": str(file_id)})
        file = data.get("file") if isinstance(data.get("file"), dict) else {}
        url = file.get("download_url")
        if not isinstance(url, str) or not url:
            raise ProviderResponseError(f"{self.name}: file {file_id} has no download URL", provider=self.name)
        response = await self._http.download(url, max_bytes=self.max_download_bytes)
        width = _int(job.metadata.get("video_width"))
        height = _int(job.metadata.get("video_height"))
        return GeneratedVideoFile(
            content=response.content, media_type=_media_type(response.content_type), model=self.model,
            reported_duration=job.request.duration, reported_width=width, reported_height=height,
            usage=VideoGenerationUsage(requests=1, seconds=job.request.duration),
            metadata={"file_id": file_id, "filename": file.get("filename"), "bytes": file.get("bytes"),
                      "resolution": job.metadata.get("resolution")})

    async def _call(self, method: str, path: str, *, json_body: dict | None = None,
                    params: dict[str, str] | None = None) -> dict:
        response = await self._http.request(method, path, json_body=json_body, params=params)
        data = response.json()
        self.check_base_resp(data, response)
        return data

    def check_base_resp(self, data: dict, response: HttpResponse | None = None) -> None:
        """MiniMax reports errors in `base_resp` with HTTP 200: map them onto the typed provider errors."""
        base = data.get("base_resp")
        if not isinstance(base, dict):
            return
        code = base.get("status_code", 0)
        if code in (0, None):
            return
        message = f"{self.name}: error {code}: {str(base.get('status_msg', ''))[:200]}"
        kwargs = {"provider": self.name, "request_id": response.vendor_request_id if response else None}
        error: ProviderError
        if code in RATE_LIMIT_CODES:
            error = ProviderRateLimit(message, **kwargs)
        elif code in AUTH_CODES:
            error = ProviderAuthenticationError(message, **kwargs)
        elif code in INVALID_CODES:
            error = ProviderInvalidRequest(message, **kwargs)
        elif code in REFUSED_CODES:
            error = ProviderResponseError(message, **kwargs)
        else:
            error = ProviderUnavailable(message, **kwargs)
        raise error


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _media_type(content_type: str) -> str:
    kind = content_type.split(";")[0].strip().lower()
    return kind if kind.startswith("video/") else "video/mp4"
