"""Video generation provider contract (MiniMax or others plug in here)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from app.schemas.common import Schema


class SceneSpec(Schema):
    scene_id: str
    narration: str
    visual_prompt: str
    duration_ms: int = Field(ge=1)


class RenderedVideo(Schema):
    content: bytes
    media_type: str
    provider: str
    duration_ms: int
    metadata: dict = Field(default_factory=dict)


class VideoProvider(ABC):
    name: str

    @abstractmethod
    async def render(self, scenes: list[SceneSpec]) -> RenderedVideo: ...
