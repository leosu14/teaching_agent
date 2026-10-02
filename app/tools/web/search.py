"""Search tool: provider-independent search returning traceable, deduplicatable search results."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.observability.scope import ExecutionScope
from app.providers.core.errors import ProviderError
from app.providers.search.base import ProviderSearchRequest, SearchHit, SearchProvider
from app.schemas.common import RetryPolicy, Schema, utcnow
from app.schemas.research import SearchQuery, SearchResult, Source
from app.tools.base import Tool, ToolError, ToolTransientError
from app.tools.research.cache import ResearchCache
from app.utils.urls import canonical_url, host_matches, source_id_for


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
                 clock: Callable[[], datetime] = utcnow, *, include_domains: list[str] | None = None,
                 exclude_domains: list[str] | None = None) -> None:
        self._provider = provider
        self._cache = cache
        self._clock = clock
        # Operator policy: which websites research may use. Sent to providers that support it, and always enforced
        # on the results, so the restriction holds whatever the provider does.
        self._include = list(include_domains or [])
        self._exclude = list(exclude_domains or [])

    def _allowed(self, url: str) -> bool:
        if self._include and not any(host_matches(url, d) for d in self._include):
            return False
        return not any(host_matches(url, d) for d in self._exclude)

    async def run(self, data: SearchQuery, scope: ExecutionScope) -> SearchResponse:
        service = f"search:{self._provider.name}"
        if self._cache is not None:
            cached = self._cache.get(data)
            if cached is not None:
                scope.usage.record_service(service=service, results=len(cached), cache_hit=True)
                return SearchResponse(query=data, results=cached, provider=self._provider.name, cached=True)
        request = ProviderSearchRequest(query=data.text, max_results=data.max_results, language=data.language,
                                        subject=data.subject, domain=data.domain, include_domains=self._include,
                                        exclude_domains=self._exclude)
        try:
            page = await self._provider.search(request)
        except ProviderError as exc:
            if exc.transient:
                raise ToolTransientError(f"search provider '{self._provider.name}' failed: {exc}") from exc
            raise ToolError(f"search provider '{self._provider.name}' failed: {exc}") from exc
        except (ConnectionError, OSError) as exc:
            raise ToolTransientError(f"search provider '{self._provider.name}' unavailable: {exc}") from exc
        retrieved_at = self._clock()
        provider = page.provider or self._provider.name  # the provider that answered, after any configured fallback
        hits = [hit for hit in page.hits if self._allowed(hit.url)]
        results = [self._result(hit, rank, retrieved_at, provider) for rank, hit in enumerate(hits, start=1)]
        scope.usage.record_service(service=f"search:{provider}", results=len(results), cost_usd=page.usage.cost_usd)
        if self._cache is not None:
            self._cache.set(data, results)
        return SearchResponse(query=data, results=results, provider=provider)

    def _result(self, hit: SearchHit, rank: int, retrieved_at: datetime, provider: str) -> SearchResult:
        source = Source(
            source_id=source_id_for(hit.url), url=hit.url, canonical_url=canonical_url(hit.url), title=hit.title,
            publisher=hit.publisher, author=hit.author, published_at=hit.published_at, retrieved_at=retrieved_at,
            language=hit.language, source_type=hit.source_type or "unknown", retrieved_via="web",
            provider=provider, metadata=hit.metadata,
        )
        return SearchResult(source_id=source.source_id, title=hit.title, url=hit.url, snippet=hit.snippet,
                            content=hit.content, rank=rank, provider_score=hit.score, source=source)
