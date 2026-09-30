"""Local knowledge-base retriever using BM25 keyword scoring. Deterministic, no external services."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from app.providers.retrieval.base import Passage, Retriever


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class LocalKnowledgeBase(Retriever):
    name = "local"

    def __init__(self, path: Path, k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = json.loads(path.read_text(encoding="utf-8"))
        self._tokens = [tokenize(f"{d['title']} {d['text']}") for d in self._docs]
        self._avg_len = sum(map(len, self._tokens)) / max(1, len(self._tokens))
        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._k1, self._b = k1, b

    def _score(self, terms: list[str], toks: list[str]) -> float:
        tf = Counter(toks)
        norm = self._k1 * (1 - self._b + self._b * len(toks) / self._avg_len)
        return sum(self._idf.get(t, 0) * tf[t] * (self._k1 + 1) / (tf[t] + norm) for t in terms if t in tf)

    async def retrieve(self, query: str, k: int, filters: dict[str, str] | None = None) -> list[Passage]:
        terms = tokenize(query)
        results = []
        for doc, toks in zip(self._docs, self._tokens):
            meta = doc.get("metadata", {})
            if filters and any(str(meta.get(key, "")).lower() != str(val).lower() for key, val in filters.items()):
                continue
            results.append(Passage(doc_id=doc["doc_id"], title=doc["title"], text=doc["text"],
                                   score=round(self._score(terms, toks), 6), metadata=meta))
        results.sort(key=lambda p: (-p.score, p.doc_id))
        return results[:k]
