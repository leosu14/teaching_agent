"""Presentation renderer contract. A renderer turns the renderer-independent Presentation into a concrete file
(PPTX now; PDF, Google Slides, ... later). It only reads the image bytes the presentation already references,
through the media reader it was given; it never searches for or generates an image, and it has no LLM cost."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import Field

from app.schemas.common import Schema
from app.schemas.presentation import ImageElement, Presentation

PPTX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

MediaReader = Callable[[str], bytes]  # object uri -> bytes


class RenderedPresentation(Schema):
    content: bytes
    media_type: str
    format: str  # file format, e.g. "pptx"
    renderer: str
    slides: int = Field(ge=1)
    elements: int = Field(ge=0)
    image_artifact_ids: list[str] = Field(default_factory=list)  # images actually placed, in order


class PresentationRenderError(Exception):
    pass


class PresentationRenderer(ABC):
    name: str
    format: str
    media_type: str

    @abstractmethod
    async def render(self, presentation: Presentation) -> RenderedPresentation: ...


def read_image(media: MediaReader, element: ImageElement) -> bytes:
    """The stored bytes of a placed image, verified against the checksum the IMAGE_ASSET recorded."""
    try:
        data = media(element.object_uri)
    except (OSError, ValueError) as exc:
        raise PresentationRenderError(f"image {element.artifact_id} cannot be read: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != element.checksum:
        raise PresentationRenderError(f"image {element.artifact_id} does not match its checksum")
    return data
