"""Presentation rendering and the PRESENTATION artifact.

The tool hands the renderer-independent Presentation to whichever PresentationRenderer is configured, checks
what came back, stores the file content-addressed in the existing object store and records a PRESENTATION
artifact whose parents are the slide plan, the lesson and the placed IMAGE_ASSET artifacts. A rendering failure
creates nothing. The same presentation renders to the same bytes, so re-running reuses the stored artifact.
"""

from __future__ import annotations

from collections import Counter

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.providers.presentation.base import PresentationRenderer, PresentationRenderError
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.events import EventType
from app.schemas.presentation import PresentationArtifactMetadata, PresentationRenderRequest
from app.tools.base import Tool, ToolError


class PresentationRenderFailed(ToolError):
    pass


class PresentationRenderTool(Tool[PresentationRenderRequest, Artifact]):
    name = "presentation.render"
    description = "Render a built presentation to a file (PPTX by default), store it in the object store and " \
                  "create the PRESENTATION artifact linked to its slide plan, lesson and images."
    input_model = PresentationRenderRequest
    output_model = Artifact
    permissions = frozenset({"artifact:write"})
    timeout_seconds = 120.0

    def __init__(self, renderer: PresentationRenderer, artifacts: ArtifactService) -> None:
        self._renderer = renderer
        self._artifacts = artifacts

    async def run(self, data: PresentationRenderRequest, scope: ExecutionScope) -> Artifact:
        if scope.task_id is None:
            raise ToolError("presentations can only be rendered inside a task")
        p = data.presentation
        scope.emit(EventType.PRESENTATION_RENDER_STARTED, tool=self.name, presentation_id=p.presentation_id,
                   renderer=self._renderer.name, format=self._renderer.format, slides=len(p.slides))
        try:
            rendered = await self._renderer.render(p)
            if rendered.slides != len(p.slides):
                raise PresentationRenderError(f"renderer produced {rendered.slides} slides for {len(p.slides)}")
            if not rendered.content:
                raise PresentationRenderError("renderer produced an empty file")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            scope.emit(EventType.PRESENTATION_FAILED, tool=self.name, stage="render", renderer=self._renderer.name,
                       presentation_id=p.presentation_id, error=error[:1000])
            raise PresentationRenderFailed(f"rendering presentation {p.presentation_id} failed: {error}") from exc

        stored = self._artifacts.put_object(rendered.content, rendered.media_type)
        scope.emit(EventType.PRESENTATION_RENDER_COMPLETED, tool=self.name, presentation_id=p.presentation_id,
                   renderer=rendered.renderer, format=rendered.format, slides=rendered.slides,
                   elements=rendered.elements, images=len(rendered.image_artifact_ids), checksum=stored.checksum,
                   size_bytes=stored.size_bytes, reused_object=stored.reused)
        scope.usage.record_service(service=f"presentation_render:{rendered.renderer}", results=1, units={
            "slides": rendered.slides, "images": len(rendered.image_artifact_ids), "bytes": stored.size_bytes})

        metadata = PresentationArtifactMetadata(
            presentation_id=p.presentation_id, deck_id=p.deck_id, title=p.title, language=p.language,
            slide_plan_artifact_id=data.slide_plan_artifact_id, lesson_artifact_id=data.lesson_artifact_id,
            renderer=rendered.renderer, format=rendered.format, media_type=rendered.media_type,
            checksum=stored.checksum, size_bytes=stored.size_bytes, object_key=stored.key or "",
            slides=rendered.slides, elements=rendered.elements, image_artifact_ids=rendered.image_artifact_ids,
            citation_ids=[r.citation_id for r in p.references],
            layouts=dict(sorted(Counter(s.layout.value for s in p.slides).items())),
        )
        previous = self._artifacts.find(scope.task_id, data.name)
        artifact = self._artifacts.store_object(
            task_id=scope.task_id, name=data.name, type=ArtifactType.PRESENTATION, obj=stored,
            provider=rendered.renderer, parent_ids=data.parent_ids, metadata=metadata.model_dump(mode="json"),
            scope=scope,
        )
        reused = previous is not None and previous.artifact_id == artifact.artifact_id
        scope.emit(EventType.PRESENTATION_ARTIFACT_CREATED, tool=self.name, artifact_id=artifact.artifact_id,
                   version=artifact.version, reused=reused, checksum=artifact.content_hash,
                   media_type=artifact.media_type, size_bytes=artifact.size_bytes, parent_ids=artifact.parent_ids)
        return artifact
