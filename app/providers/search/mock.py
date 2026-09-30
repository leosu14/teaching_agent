"""Deterministic web search over a local JSON corpus (stands in for a real search API)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.providers.search.base import SearchHit, SearchProvider


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class CorpusSearchProvider(SearchProvider):
    name = "mock"

    def __init__(self, corpus_path: Path) -> None:
        self._docs = json.loads(corpus_path.read_text(encoding="utf-8"))

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        terms = set(tokenize(query))
        scored = []
        for doc in self._docs:
            haystack = tokenize(f"{doc['title']} {doc['snippet']} {doc.get('text', '')}")
            score = sum(1 for tok in haystack if tok in terms)
            if score:
                scored.append((-score, doc["url"], doc))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [
            SearchHit(
                url=doc["url"],
                title=doc["title"],
                snippet=doc["snippet"],
                publisher=doc["publisher"],
                metadata={k: v for k, v in doc.items() if k not in {"url", "title", "snippet", "publisher"}},
            )
            for _, _, doc in scored[:max_results]
        ]
