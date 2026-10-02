"""Deterministic image search over a local JSON catalogue. It stands in for a real image API: no network access.

Catalogue entries use the ImageHit field names; any other keys are passed through as metadata. Downloads
return a PNG rendered locally from the entry, so the bytes are deterministic. A `mock_actual_size` key makes
the downloaded image differ from the advertised size, like a provider whose metadata is wrong.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.providers.core.errors import ProviderError
from app.providers.image_search.base import (
    DownloadedImage,
    ImageHit,
    ImageSearchPage,
    ImageSearchProvider,
    ProviderImageSearchRequest,
)
from app.schemas.visual import ImageUsage
from app.utils.images import colors_from, encode_png

HIT_FIELDS = set(ImageHit.model_fields) - {"score", "metadata"}
STOPWORDS = frozenset("a an and the of in on for with to by at is are about from".split())


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS}


class MockImageSearchProvider(ImageSearchProvider):
    name = "mock"

    def __init__(self, catalog_path: Path) -> None:
        self._entries = json.loads(catalog_path.read_text(encoding="utf-8"))
        self._by_id = {e["provider_image_id"]: e for e in self._entries}

    async def search(self, request: ProviderImageSearchRequest) -> ImageSearchPage:
        terms = tokenize(request.query)
        scored = []
        for entry in self._entries:
            words = tokenize(" ".join([entry["title"], entry.get("description", ""), *entry.get("tags", [])]))
            score = len(terms & words)
            if score:
                scored.append((-score, entry["provider_image_id"], entry))
        scored.sort(key=lambda item: (item[0], item[1]))
        hits = [
            ImageHit(score=float(-neg), metadata={k: v for k, v in e.items() if k not in HIT_FIELDS},
                     **{k: v for k, v in e.items() if k in HIT_FIELDS})
            for neg, _, e in scored[:request.max_results]
        ]
        return ImageSearchPage(hits=hits, usage=ImageUsage(requests=1, results=len(hits), cost_usd=0.0))

    async def download(self, provider_image_id: str, url: str) -> DownloadedImage:
        entry = self._by_id.get(provider_image_id)
        if entry is None or entry["url"] != url:
            raise ProviderError(f"no image {provider_image_id} at {url}", transient=False)
        width, height = entry.get("mock_actual_size") or (entry["width"], entry["height"])
        digest = hashlib.sha256(f"{self.name}:{provider_image_id}".encode()).digest()
        return DownloadedImage(content=encode_png(width, height, colors_from(digest, 3)), media_type="image/png")
