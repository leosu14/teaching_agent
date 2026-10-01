"""Image search provider contract. Adapters (Unsplash, Wikimedia Commons, Pexels, ...) map their API to these
schemas. A provider reports only what its API returns: unknown fields stay None, never guessed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from app.schemas.common import Schema
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
    metadata: dict = Field(default_factory=dict)  # any other provider fields, verbatim


class ImageSearchPage(Schema):
    hits: list[ImageHit]
    usage: ImageUsage


class DownloadedImage(Schema):
    content: bytes
    media_type: str


class ImageSearchProviderError(Exception):
    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class ImageSearchProvider(ABC):
    name: str

    @abstractmethod
    async def search(self, request: ProviderImageSearchRequest) -> ImageSearchPage: ...

    @abstractmethod
    async def download(self, provider_image_id: str, url: str) -> DownloadedImage: ...
