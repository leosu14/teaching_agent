"""Image search provider contract. Adapters (Unsplash, Wikimedia Commons, Pexels, ...) map their API to these
schemas. A provider reports only what its API returns: unknown fields stay None, never guessed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from pydantic import Field

from app.providers.core.base import Provider
from app.schemas.common import Schema
from app.schemas.providers import Capability
from app.schemas.visual import ImageUsage


class ProviderImageSearchRequest(Schema):
    query: str = Field(min_length=1)
    max_results: int = Field(ge=1, le=50)
    kind: str | None = None  # visual type hint, e.g. "photo"
    orientation: str | None = None  # "landscape", "portrait" or "square"
    language: str | None = None


class ImageHit(Schema):
    provider_image_id: str
    url: str
    title: str
    source_url: str
    width: int
    height: int
    format: str  # MIME type
    thumbnail_url: str | None = None
    description: str = ""
    publisher: str | None = None
    creator: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution_text: str | None = None
    kind: str | None = None
    source_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    score: float | None = None
    # "searched": the image is in the provider's own library, with its licence data; "external": the provider only
    # indexes an image hosted by a third party, so its licence and attribution are whatever the source states.
    origin: Literal["searched", "external"] = "searched"
    metadata: dict = Field(default_factory=dict)  # any other provider fields, verbatim


class ImageSearchPage(Schema):
    hits: list[ImageHit]
    usage: ImageUsage


class DownloadedImage(Schema):
    content: bytes
    media_type: str


class ImageSearchProvider(Provider, ABC):
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.IMAGE_SEARCH})

    @abstractmethod
    async def search(self, request: ProviderImageSearchRequest) -> ImageSearchPage: ...

    @abstractmethod
    async def download(self, provider_image_id: str, url: str) -> DownloadedImage: ...
