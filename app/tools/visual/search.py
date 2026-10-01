"""Image search and fetch tools: provider-independent candidates with stable ids and verbatim source metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.providers.image_search.base import (
    ImageHit,
    ImageSearchProvider,
    ImageSearchProviderError,
    ProviderImageSearchRequest,
)
from app.schemas.common import RetryPolicy, utcnow
from app.schemas.visual import (
    FetchedImage,
    ImageFetchRequest,
    ImageLicense,
    ImageSearchRequest,
    ImageSearchResponse,
    ImageSearchResult,
    VisualType,
    aspect_value,
)
from app.tools.base import Tool, ToolError, ToolTransientError


def image_id_for(provider: str, provider_image_id: str) -> str:
    """Stable id of a provider's image: the same image always gets the same id."""
    return "imgsrc_" + hashlib.sha256(f"{provider}:{provider_image_id}".encode()).hexdigest()[:16]


def _orientation(aspect_ratio: str | None) -> str | None:
    if aspect_ratio is None:
        return None
    ratio = aspect_value(aspect_ratio)
    return "landscape" if ratio > 1.05 else "portrait" if ratio < 0.95 else "square"


def _provider_error(provider: str, exc: Exception) -> ToolError:
    transient = getattr(exc, "transient", isinstance(exc, (ConnectionError, OSError)))
    cls = ToolTransientError if transient else ToolError
    return cls(f"image search provider '{provider}' failed: {exc}")


class ImageSearchTool(Tool[ImageSearchRequest, ImageSearchResponse]):
    name = "image.search"
    description = "Search images through the configured provider; candidates carry stable ids, size, format, " \
                  "licence and attribution exactly as the provider reported them."
    input_model = ImageSearchRequest
    output_model = ImageSearchResponse
    permissions = frozenset({"network"})
    timeout_seconds = 20.0
    retry = RetryPolicy(max_attempts=3, backoff_seconds=0.2)

    def __init__(self, provider: ImageSearchProvider, clock: Callable[[], datetime] = utcnow) -> None:
        self._provider = provider
        self._clock = clock

    async def run(self, data: ImageSearchRequest, scope: ExecutionScope) -> ImageSearchResponse:
        request = ProviderImageSearchRequest(
            query=data.query, max_results=data.max_results, language=data.language,
            kind=data.visual_type.value if data.visual_type else None, orientation=_orientation(data.aspect_ratio),
        )
        try:
            page = await self._provider.search(request)
        except (ImageSearchProviderError, ConnectionError, OSError) as exc:
            raise _provider_error(self._provider.name, exc) from exc
        retrieved_at = self._clock()
        results = [self._result(hit, rank, retrieved_at) for rank, hit in enumerate(page.hits, start=1)]
        scope.usage.record_service(service=f"image_search:{self._provider.name}", results=len(results),
                                   cost_usd=page.usage.cost_usd, units={"requests": page.usage.requests})
        return ImageSearchResponse(request=data, results=results, provider=self._provider.name, usage=page.usage)

    def _result(self, hit: ImageHit, rank: int, retrieved_at: datetime) -> ImageSearchResult:
        metadata = dict(hit.metadata)
        if hit.kind is not None:
            metadata.setdefault("provider_kind", hit.kind)
        if hit.score is not None:
            metadata.setdefault("provider_score", hit.score)
        return ImageSearchResult(
            image_id=image_id_for(self._provider.name, hit.provider_image_id), provider=self._provider.name,
            provider_image_id=hit.provider_image_id, url=hit.url, thumbnail_url=hit.thumbnail_url, title=hit.title,
            description=hit.description, source_url=hit.source_url, publisher=hit.publisher, creator=hit.creator,
            width=hit.width, height=hit.height, format=hit.format,
            license=ImageLicense(name=hit.license_name, url=hit.license_url) if hit.license_name else None,
            attribution_text=hit.attribution_text, visual_type=VisualType.parse(hit.kind),
            source_type=hit.source_type or "unknown", tags=hit.tags, rank=rank, retrieved_at=retrieved_at,
            metadata=metadata,
        )


class ImageFetchTool(Tool[ImageFetchRequest, FetchedImage]):
    name = "image.fetch"
    description = "Download a searched image through its provider into the object store (content-addressed: " \
                  "identical bytes are stored once)."
    input_model = ImageFetchRequest
    output_model = FetchedImage
    permissions = frozenset({"network", "artifact:write"})
    timeout_seconds = 30.0
    retry = RetryPolicy(max_attempts=3, backoff_seconds=0.2)

    def __init__(self, provider: ImageSearchProvider, artifacts: ArtifactService) -> None:
        self._provider = provider
        self._artifacts = artifacts

    async def run(self, data: ImageFetchRequest, scope: ExecutionScope) -> FetchedImage:
        result = data.result
        if result.provider != self._provider.name:
            raise ToolError(f"image {result.image_id} comes from provider '{result.provider}', "
                            f"not '{self._provider.name}'")
        try:
            image = await self._provider.download(result.provider_image_id, result.url)
        except (ImageSearchProviderError, ConnectionError, OSError) as exc:
            raise _provider_error(self._provider.name, exc) from exc
        if not image.content:
            raise ToolError(f"image {result.image_id} downloaded empty")
        obj = self._artifacts.put_object(image.content, image.media_type)
        scope.usage.record_service(service=f"image_fetch:{self._provider.name}", results=1,
                                   units={"bytes": obj.size_bytes})
        return FetchedImage(image_id=result.image_id, object=obj)
