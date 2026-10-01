"""Video tool. Results are stored as artifacts so agents never handle files or URLs directly. Not used by the lesson
workflow yet. Image tools live in app/tools/visual/, speech and audio tools in app/tools/audio/."""

from __future__ import annotations

from pydantic import Field

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.providers.video.base import SceneSpec, VideoProvider
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.common import Schema
from app.tools.base import Tool, ToolError


def _require_task(scope: ExecutionScope) -> str:
    if scope.task_id is None:
        raise ToolError("media artifacts can only be created inside a task")
    return scope.task_id


class VideoInput(Schema):
    name: str
    scenes: list[SceneSpec] = Field(min_length=1)
    parent_ids: list[str] = Field(default_factory=list)


class VideoRenderTool(Tool[VideoInput, Artifact]):
    name = "video.render"
    description = "Render scenes into a video and store it as a VIDEO artifact."
    input_model = VideoInput
    output_model = Artifact
    permissions = frozenset({"media:generate", "artifact:write"})
    timeout_seconds = 600.0

    def __init__(self, provider: VideoProvider, artifacts: ArtifactService) -> None:
        self._provider = provider
        self._artifacts = artifacts

    async def run(self, data: VideoInput, scope: ExecutionScope) -> Artifact:
        video = await self._provider.render(data.scenes)
        return self._artifacts.store(
            task_id=_require_task(scope), name=data.name, type=ArtifactType.VIDEO, media_type=video.media_type,
            content=video.content, provider=video.provider, parent_ids=data.parent_ids,
            metadata={**video.metadata, "duration_ms": video.duration_ms}, scope=scope,
        )
