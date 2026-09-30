"""Artifact storage tool."""

from __future__ import annotations

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.schemas.artifact import ArtifactBatch, StoredArtifacts
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
