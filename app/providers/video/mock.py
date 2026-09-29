"""Deterministic video provider: returns a render manifest instead of encoded video."""

from __future__ import annotations

import json

from app.providers.video.base import RenderedVideo, SceneSpec, VideoProvider


class MockVideoProvider(VideoProvider):
    name = "mock"

    async def render(self, scenes: list[SceneSpec]) -> RenderedVideo:
        manifest = {"scenes": [s.model_dump() for s in scenes]}
        return RenderedVideo(
            content=json.dumps(manifest, ensure_ascii=False).encode(),
            media_type="application/vnd.teaching-agent.video-manifest+json",
            provider=self.name,
            duration_ms=sum(s.duration_ms for s in scenes),
            metadata={"scene_count": len(scenes)},
        )
