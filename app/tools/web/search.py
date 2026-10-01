"""Search tool: provider-independent search returning traceable, deduplicatable search results."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.observability.scope import ExecutionScope
from app.providers.search.base import ProviderSearchRequest, SearchHit, SearchProvider, SearchProviderError
from app.schemas.common import RetryPolicy, Schema, utcnow
from app.schemas.research import SearchQuery, SearchResult, Source
from app.tools.base import Tool, ToolError, ToolTransientError
from app.tools.research.cache import ResearchCache
from app.utils.urls import canonical_url, source_id_for


class SearchResponse(Schema):
    query: SearchQuery
    results: list[SearchResult]
    provider: str
    cached: bool = False


class SearchTool(Tool[SearchQuery, SearchResponse]):
    name = "search.web"
    description = "Search the web through the configured provider; results carry stable source ids and metadata."
    input_model = SearchQuery
    output_model = SearchResponse
    permissions = frozenset({"network"})
    timeout_seconds = 20.0
    retry = RetryPolicy(max_attempts=3, backoff_seconds=0.2)

    def __init__(self, provider: SearchProvider, cache: ResearchCache | None = None,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self._provider = provider
        self._cache = cache
        self._clock = clock

    async def run(self, data: SearchQuery, scope: ExecutionScope) -> SearchResponse:
        service = f"search:{self._provider.name}"
        if self._cache is not None:
            cached = self._cache.get(data)
            if cached is not None:
                scope.usage.record_service(service=service, results=len(cached), cache_hit=True)
                return SearchResponse(query=data, results=cached, provider=self._provider.name, cached=True)
        request = ProviderSearchRequest(query=data.text, max_results=data.max_results, language=data.language,
                                        subject=data.subject, domain=data.domain)
        try:
            page = await self._provider.search(request)
        except SearchProviderError as exc:
            if exc.transient:
                raise ToolTransientError(f"search provider '{self._provider.name}' failed: {exc}") from exc
            raise ToolError(f"search provider '{self._provider.name}' failed: {exc}") from exc
        except (ConnectionError, OSError) as exc:
            raise ToolTransientError(f"search provider '{self._provider.name}' unavailable: {exc}") from exc
        retrieved_at = self._clock()
        results = [self._result(hit, rank, retrieved_at) for rank, hit in enumerate(page.hits, start=1)]
        scope.usage.record_service(service=service, results=len(results), cost_usd=page.usage.cost_usd)
        if self._cache is not None:
            self._cache.set(data, results)
        return SearchResponse(query=data, results=results, provider=self._provider.name)

    def _result(self, hit: SearchHit, rank: int, retrieved_at: datetime) -> SearchResult:
        source = Source(
            source_id=source_id_for(hit.url), url=hit.url, canonical_url=canonical_url(hit.url), title=hit.title,
            publisher=hit.publisher, author=hit.author, published_at=hit.published_at, retrieved_at=retrieved_at,
            language=hit.language, source_type=hit.source_type or "unknown", retrieved_via="web",
            provider=self._provider.name, metadata=hit.metadata,
        )
        return SearchResult(source_id=source.source_id, title=hit.title, url=hit.url, snippet=hit.snippet,
                            content=hit.content, rank=rank, provider_score=hit.score, source=source)
