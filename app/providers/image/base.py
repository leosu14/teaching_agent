"""Image generation provider contract. Adapters (MiniMax, OpenAI Images, Stability, ...) map their API to
these schemas. A provider declares which optional parameters it supports; the tool records the rest as ignored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from pydantic import Field

from app.providers.core.base import Provider
from app.schemas.common import Schema
from app.schemas.providers import Capability
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
    # A generation provider only ever produces generated images; searched and externally sourced images come from
    # ImageSearchProvider, with the source's own attribution.
    origin: Literal["generated"] = "generated"
    provider: str | None = None  # set by the provider layer: the provider that actually produced it (fallback)


class ImageGenerationProvider(Provider, ABC):
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.IMAGE})
    supports_negative_prompt: bool = False
    supports_seed: bool = False
    supports_style: bool = False

    @abstractmethod
    async def generate(self, request: ProviderImageRequest) -> GeneratedImage: ...
