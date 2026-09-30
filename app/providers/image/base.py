"""Image generation provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from app.schemas.common import Schema


class GeneratedImage(Schema):
    content: bytes
    media_type: str
    provider: str
    metadata: dict = Field(default_factory=dict)


class ImageProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, prompt: str, width: int, height: int) -> GeneratedImage: ...
