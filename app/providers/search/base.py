"""Web search provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from app.schemas.common import Schema


class SearchHit(Schema):
    url: str
    title: str
    snippet: str
    publisher: str
    metadata: dict = Field(default_factory=dict)


class SearchProvider(ABC):
    name: str

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...
