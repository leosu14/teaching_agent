"""Deterministic image selection: candidates are scored on fixed signals, never by a model.

Signals (each 0..1): relevance to the query, closeness to the expected aspect ratio, visual-type match,
licence/attribution availability, source quality by provider-reported source type, and resolution. A
candidate without a licence is ineligible when attribution is required; an irrelevant one is ineligible.
Ties break on the provider's rank, then the image id, so the same input always gives the same order.
"""

from __future__ import annotations

import math
import re

from app.observability.scope import ExecutionScope
from app.schemas.visual import (
    ImageSearchResult,
    ImageSelectionRequest,
    ImageSelectionResult,
    RankedImage,
    SelectionSignals,
    VisualRequirement,
    VisualType,
    aspect_value,
)
from app.tools.base import Tool

STOPWORDS = frozenset("a an and the of in on for with to by at is are about from".split())
SOURCE_QUALITY = {"media_archive": 1.0, "museum": 1.0, "encyclopedia": 0.9, "illustration_library": 0.8,
                  "icon_library": 0.8, "stock": 0.7, "user_upload": 0.3}
COMPATIBLE = {frozenset(pair) for pair in (
    (VisualType.PHOTO, VisualType.ILLUSTRATION), (VisualType.ILLUSTRATION, VisualType.ICON),
    (VisualType.DIAGRAM, VisualType.CHART), (VisualType.DIAGRAM, VisualType.ILLUSTRATION),
)}
GOOD_SHORT_SIDE = 720


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS}


class ImageSelector:
    name = "deterministic-v1"

    def signals(self, req: VisualRequirement, c: ImageSearchResult) -> SelectionSignals:
        query = _terms(req.search_query or req.description)
        words = _terms(" ".join([c.title, c.description, *c.tags]))
        relevance = len(query & words) / len(query) if query else 0.0
        if req.aspect_ratio:
            off = abs(math.log((c.width / c.height) / aspect_value(req.aspect_ratio)))
            aspect = max(0.0, 1 - off / math.log(2))
        else:
            aspect = 1.0
        if c.visual_type is None:
            vtype = 0.5
        elif c.visual_type == req.visual_type:
            vtype = 1.0
        else:
            vtype = 0.5 if frozenset({c.visual_type, req.visual_type}) in COMPATIBLE else 0.0
        if c.license and (c.creator or c.attribution_text):
            lic = 1.0
        elif c.license:
            lic = 0.6
        else:
            lic = 0.0
        return SelectionSignals(
            relevance=round(relevance, 4), aspect=round(aspect, 4), visual_type=vtype, license=lic,
            source_quality=SOURCE_QUALITY.get(c.source_type, 0.5),
            resolution=round(min(1.0, min(c.width, c.height) / GOOD_SHORT_SIDE), 4),
        )

    def rank(self, request: ImageSelectionRequest) -> ImageSelectionResult:
        req, w = request.requirement, request.weights
        total = sum(w.model_dump().values()) or 1.0
        scored = []
        for c in request.candidates:
            s = self.signals(req, c)
            score = round(sum(getattr(w, k) * v for k, v in s.model_dump().items()) / total, 4)
            reasons = []
            if s.relevance == 0:
                reasons.append("not relevant to the search query")
            if req.attribution_required and c.license is None:
                reasons.append("no licence information, and this visual requires attribution")
            scored.append((not reasons, score, s, reasons, c))
        scored.sort(key=lambda x: (not x[0], -x[1], x[4].rank, x[4].image_id))
        return ImageSelectionResult(selector=self.name, ranked=[
            RankedImage(rank=i, score=score, eligible=eligible, reasons=reasons, signals=s, candidate=c)
            for i, (eligible, score, s, reasons, c) in enumerate(scored, start=1)
        ])


class ImageSelectionTool(Tool[ImageSelectionRequest, ImageSelectionResult]):
    name = "image.select"
    description = "Rank image candidates for a visual requirement with a deterministic score (relevance, aspect " \
                  "ratio, visual type, licence, source quality, resolution); candidates keep all their metadata."
    input_model = ImageSelectionRequest
    output_model = ImageSelectionResult

    def __init__(self, selector: ImageSelector | None = None) -> None:
        self._selector = selector or ImageSelector()

    async def run(self, data: ImageSelectionRequest, scope: ExecutionScope) -> ImageSelectionResult:
        return self._selector.rank(data)
