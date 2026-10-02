"""Search provider contract. Adapters (Tavily, Serper, Bing, Google, ...) map their API to these schemas.

A provider reports only what its API returns: optional fields stay None rather than being guessed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import ClassVar

from pydantic import Field

from app.providers.core.base import Provider
from app.schemas.common import Schema
from app.schemas.providers import Capability


class ProviderSearchRequest(Schema):
    query: str = Field(min_length=1)
    max_results: int = Field(ge=1, le=50)
    language: str | None = None
    subject: str | None = None
    domain: str | None = None  # the knowledge domain of the query (a search hint), not a website
    include_domains: list[str] = Field(default_factory=list)  # restrict results to these websites
    exclude_domains: list[str] = Field(default_factory=list)


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
    provider: str | None = None  # set by the provider layer: the provider that actually answered (fallback)


class SearchProvider(Provider, ABC):
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.SEARCH})
    supports_domain_filter: ClassVar[bool] = False  # when False, the search tool filters results by host itself

    @abstractmethod
    async def search(self, request: ProviderSearchRequest) -> SearchPage: ...
