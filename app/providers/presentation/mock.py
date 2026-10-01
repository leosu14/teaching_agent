"""Deterministic renderer for tests: a canonical JSON description of what a real renderer would draw."""

from __future__ import annotations

import json

from app.providers.presentation.base import PresentationRenderer, RenderedPresentation
from app.schemas.presentation import Presentation

MEDIA_TYPE = "application/json"


def _element(e) -> dict:
    out = {"id": e.element_id, "kind": e.kind, "region": e.region}
    match e.kind:
        case "title" | "text" | "footer":
            out["text"] = e.text
        case "bullets":
            out["items"] = e.items
        case "table":
            out["headers"], out["rows"] = e.headers, e.rows
        case "image":
            out.update(artifact_id=e.artifact_id, asset_id=e.asset_id, checksum=e.checksum)
    if e.kind == "footer":
        out.update(citation_ids=e.citation_ids, image_artifact_ids=e.image_artifact_ids)
    return out


class MockPresentationRenderer(PresentationRenderer):
    name = "mock-presentation"
    format = "json"
    media_type = MEDIA_TYPE

    async def render(self, presentation: Presentation) -> RenderedPresentation:
        doc = {
            "renderer": self.name, "presentation_id": presentation.presentation_id, "deck_id": presentation.deck_id,
            "title": presentation.title, "aspect_ratio": presentation.config.aspect_ratio,
            "slides": [{"order": s.order, "slide_id": s.slide_id, "slide_type": s.slide_type.value,
                        "layout": s.layout.value, "elements": [_element(e) for e in s.elements],
                        "citation_ids": s.citation_ids, "image_artifact_ids": s.image_artifact_ids,
                        "notes": s.notes} for s in presentation.slides],
            "references": [r.model_dump(mode="json") for r in presentation.references],
        }
        return RenderedPresentation(
            content=json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False).encode("utf-8"),
            media_type=self.media_type, format=self.format, renderer=self.name, slides=len(presentation.slides),
            elements=presentation.element_count(), image_artifact_ids=presentation.image_artifact_ids(),
        )
