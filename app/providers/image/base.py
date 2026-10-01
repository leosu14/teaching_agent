"""Image generation provider contract. Adapters (MiniMax, OpenAI Images, Stability, ...) map their API to
these schemas. A provider declares which optional parameters it supports; the tool records the rest as ignored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from app.schemas.common import Schema
from app.schemas.visual import ImageUsage


class ProviderImageRequest(Schema):
    prompt: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    negative_prompt: str | None = None
    style: str | None = None
    seed: int | None = None


class GeneratedImage(Schema):
    content: bytes
    media_type: str
    width: int
    height: int
    model: str
    seed: int | None = None
    usage: ImageUsage
    metadata: dict = Field(default_factory=dict)  # anything else the provider returned, verbatim


class ImageGenerationProviderError(Exception):
    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class ImageGenerationProvider(ABC):
    name: str
    supports_negative_prompt: bool = False
    supports_seed: bool = False
    supports_style: bool = False

    @abstractmethod
    async def generate(self, request: ProviderImageRequest) -> GeneratedImage: ...
