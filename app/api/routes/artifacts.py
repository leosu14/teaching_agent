"""Artifact content. Ownership is checked first (`owned_artifact`), then the content policy
(`app/services/artifact_content.py`): an artifact of another learner, an unknown id and a type that is not servable
all get the same 404. Served content is streamed as a download with no storage location, digest or length."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.authorization import owned_artifact
from app.api.deps import container
from app.services.container import Container

router = APIRouter(prefix="/artifacts", tags=["artifacts"])

# No Content-Length, ETag or Last-Modified: a byte length or digest could check a guessed answer key (see ArtifactView).
CONTENT_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}


@router.get("/{artifact_id}/content", response_class=StreamingResponse, dependencies=[Depends(owned_artifact)],
            responses={200: {"content": {"application/octet-stream": {}}, "description": "The artifact's content"}})
def artifact_content(artifact_id: str, c: Container = Depends(container)) -> StreamingResponse:
    content = c.artifact_content.open(artifact_id)
    return StreamingResponse(content.chunks(), media_type=content.media_type, headers={
        **CONTENT_HEADERS, "Content-Disposition": f'attachment; filename="{content.filename}"'})
