"""Deterministic search over a local JSON corpus. It stands in for a real search API: no network access."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.providers.search.base import ProviderSearchRequest, SearchHit, SearchPage, SearchProvider, SearchUsage
from app.utils.urls import host_matches

HIT_FIELDS = set(SearchHit.model_fields) - {"score", "metadata"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class MockSearchProvider(SearchProvider):
    """Scores documents by query-term frequency and returns them in a stable order.

    Corpus entries use the SearchHit field names; any other keys are passed through as metadata.
    Usage is one request per search with zero cost, since no paid API is called.
    """

    name = "mock"
    supports_domain_filter = True

    def __init__(self, corpus_path: Path) -> None:
        self._docs = json.loads(corpus_path.read_text(encoding="utf-8"))

    async def search(self, request: ProviderSearchRequest) -> SearchPage:
        terms = set(tokenize(" ".join(filter(None, [request.query, request.subject, request.domain]))))
        scored = []
        for doc in self._docs:
            if request.language and doc.get("language") not in (None, request.language):
                continue
            if request.include_domains and not any(host_matches(doc["url"], d) for d in request.include_domains):
                continue
            if any(host_matches(doc["url"], d) for d in request.exclude_domains):
                continue
            haystack = tokenize(f"{doc['title']} {doc['snippet']} {doc.get('content') or ''}")
            score = sum(1 for tok in haystack if tok in terms)
            if score:
                scored.append((-score, doc["url"], doc))
        scored.sort(key=lambda item: (item[0], item[1]))
        hits = [
            SearchHit(score=float(-neg), metadata={k: v for k, v in doc.items() if k not in HIT_FIELDS},
                      **{k: v for k, v in doc.items() if k in HIT_FIELDS})
            for neg, _, doc in scored[:request.max_results]
        ]
        return SearchPage(hits=hits, usage=SearchUsage(requests=1, results=len(hits), cost_usd=0.0))
