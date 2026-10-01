"""Research cache. Entries hold complete search results (sources, retrieval times, provider), so a
cached answer is exactly as traceable as a fresh one."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.research import SearchQuery, SearchResult


class ResearchCache(ABC):
    @abstractmethod
    def get(self, query: SearchQuery) -> list[SearchResult] | None: ...

    @abstractmethod
    def set(self, query: SearchQuery, results: list[SearchResult]) -> None: ...

    @abstractmethod
    def invalidate(self, query: SearchQuery) -> None: ...


class InMemoryResearchCache(ResearchCache):
    """Process-local cache keyed by `SearchQuery.cache_key()`. Stores deep copies so callers cannot mutate it."""

    def __init__(self) -> None:
        self._entries: dict[str, list[SearchResult]] = {}

    def get(self, query: SearchQuery) -> list[SearchResult] | None:
        hit = self._entries.get(query.cache_key())
        return None if hit is None else [r.model_copy(deep=True) for r in hit]

    def set(self, query: SearchQuery, results: list[SearchResult]) -> None:
        self._entries[query.cache_key()] = [r.model_copy(deep=True) for r in results]

    def invalidate(self, query: SearchQuery) -> None:
        self._entries.pop(query.cache_key(), None)

    def __len__(self) -> int:
        return len(self._entries)
