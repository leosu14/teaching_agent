"""Search provider contract. Adapters (Tavily, Serper, Bing, Google, ...) map their API to these schemas.

A provider reports only what its API returns: optional fields stay None rather than being guessed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from pydantic import Field

from app.schemas.common import Schema


class ProviderSearchRequest(Schema):
    query: str = Field(min_length=1)
    max_results: int = Field(ge=1, le=50)
    language: str | None = None
    subject: str | None = None
    domain: str | None = None


class SearchHit(Schema):
    url: str
    title: str
    snippet: str
    content: str | None = None
    publisher: str | None = None
    author: str | None = None
    published_at: date | None = None
    language: str | None = None
    source_type: str | None = None
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class SearchUsage(Schema):
    """Deterministic usage units. `cost_usd` is set only when the provider itself reports a cost."""

    requests: int = Field(default=1, ge=0)
    results: int = Field(default=0, ge=0)
    cost_usd: float | None = None


class SearchPage(Schema):
    hits: list[SearchHit]
    usage: SearchUsage


class SearchProviderError(Exception):
    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class SearchProvider(ABC):
    name: str

    @abstractmethod
    async def search(self, request: ProviderSearchRequest) -> SearchPage: ...
