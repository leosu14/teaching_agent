"""Web search adapter for the Tavily Search API (POST /search). Plain HTTP: no vendor SDK.

Hits carry only what Tavily returned: `snippet` is its extracted content, `content` the full page text when
`include_raw_content` is on (research checks evidence quotes against it). Publisher, author and language stay None,
since the API does not report them; a published date is kept only when the API gives one.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import ClassVar

from app.providers.core.errors import ProviderResponseError
from app.providers.core.http import HttpClient
from app.providers.search.base import ProviderSearchRequest, SearchHit, SearchPage, SearchProvider, SearchUsage

MAX_RESULTS = 20  # the API's limit per request


def _date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    for parse in (lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")).date(),
                  lambda v: datetime.strptime(v, "%a, %d %b %Y %H:%M:%S %Z").date()):
        try:
            return parse(value)
        except ValueError:
            continue
    return None


class TavilySearchProvider(SearchProvider):
    name = "tavily"
    requires_network: ClassVar[bool] = True
    supports_domain_filter: ClassVar[bool] = True

    def __init__(self, http: HttpClient, *, search_depth: str = "basic", include_raw_content: bool = True) -> None:
        self._http = http
        self._depth = search_depth
        self._raw = include_raw_content

    def configuration(self) -> dict:
        return {"base_url": self._http.base_url, "search_depth": self._depth, "include_raw_content": self._raw}

    async def probe(self) -> str:
        return "local"  # the API has no free endpoint: a probe would spend credits, so only config is checked

    async def search(self, request: ProviderSearchRequest) -> SearchPage:
        body: dict = {"query": request.query, "max_results": min(request.max_results, MAX_RESULTS),
                      "search_depth": self._depth, "include_raw_content": self._raw, "include_answer": False,
                      "include_images": False}
        if request.include_domains:
            body["include_domains"] = request.include_domains
        if request.exclude_domains:
            body["exclude_domains"] = request.exclude_domains
        response = await self._http.request("POST", "/search", json_body=body)
        data = response.json()
        results = data.get("results")
        if not isinstance(results, list):
            raise ProviderResponseError(f"{self.name}: response has no results list", provider=self.name)
        hits = []
        for r in results:
            if not isinstance(r, dict) or not r.get("url") or not r.get("title"):
                continue  # an unusable hit is dropped, never completed with guesses
            metadata = {k: r[k] for k in ("favicon",) if r.get(k)}
            if response.vendor_request_id or data.get("request_id"):
                metadata["vendor_request_id"] = response.vendor_request_id or data.get("request_id")
            hits.append(SearchHit(
                url=r["url"], title=r["title"], snippet=r.get("content") or "", content=r.get("raw_content") or None,
                published_at=_date(r.get("published_date")), score=r.get("score"), metadata=metadata,
            ))
        return SearchPage(hits=hits, usage=SearchUsage(requests=1, results=len(hits)))
