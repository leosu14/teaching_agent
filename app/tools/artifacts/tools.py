"""Artifact storage tool."""

from __future__ import annotations

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from pydantic import Field

from app.schemas.artifact import Artifact, ArtifactBatch, StoredArtifacts
from app.schemas.common import Schema
from app.tools.base import Tool, ToolError


class StoreArtifactsTool(Tool[ArtifactBatch, StoredArtifacts]):
    name = "artifact.store"
    description = "Store a batch of artifacts with versioning and parent links."
    input_model = ArtifactBatch
    output_model = StoredArtifacts
    permissions = frozenset({"artifact:write"})

    def __init__(self, artifacts: ArtifactService) -> None:
        self._artifacts = artifacts

    async def run(self, data: ArtifactBatch, scope: ExecutionScope) -> StoredArtifacts:
        if scope.task_id is None:
            raise ToolError("artifacts can only be stored inside a task")
        return self._artifacts.store_batch(scope.task_id, data.drafts, scope)


class ReadArtifactsInput(Schema):
    task_id: str
    names: list[str] = Field(min_length=1)


class ArtifactContent(Schema):
    artifact: Artifact
    content: str


class ReadArtifactsOutput(Schema):
    items: list[ArtifactContent]

    def content(self, name: str) -> str:
        return next(i.content for i in self.items if i.artifact.name == name)

    def artifact(self, name: str) -> Artifact:
        return next(i.artifact for i in self.items if i.artifact.name == name)


class ReadArtifactsTool(Tool[ReadArtifactsInput, ReadArtifactsOutput]):
    name = "artifact.read"
    description = "Read the latest version of named text artifacts of a task."
    input_model = ReadArtifactsInput
    output_model = ReadArtifactsOutput
    permissions = frozenset({"artifact:read"})

    def __init__(self, artifacts: ArtifactService) -> None:
        self._artifacts = artifacts

    async def run(self, data: ReadArtifactsInput, scope: ExecutionScope) -> ReadArtifactsOutput:
        items = []
        for name in data.names:
            artifact = self._artifacts.find(data.task_id, name)
            if artifact is None:
                raise ToolError(f"task {data.task_id} has no artifact named '{name}'")
            items.append(ArtifactContent(artifact=artifact,
                                         content=self._artifacts.read(artifact.artifact_id).decode("utf-8")))
        return ReadArtifactsOutput(items=items)
