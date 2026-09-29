"""Web search tool: provider-independent search returning traceable source candidates."""

from __future__ import annotations

import hashlib

from pydantic import Field

from app.observability.scope import ExecutionScope
from app.providers.search.base import SearchProvider
from app.schemas.common import RetryPolicy, Schema
from app.schemas.lesson import SourceCandidate
from app.tools.base import Tool, ToolTransientError


class WebSearchInput(Schema):
    query: str = Field(min_length=1)
    max_results: int = Field(default=8, ge=1, le=50)


class WebSearchOutput(Schema):
    candidates: list[SourceCandidate]


def source_id_for(uri: str) -> str:
    return "src_" + hashlib.sha256(uri.encode()).hexdigest()[:10]


class WebSearchTool(Tool[WebSearchInput, WebSearchOutput]):
    name = "search.web"
    description = "Search the web and return source candidates with stable source ids."
    input_model = WebSearchInput
    output_model = WebSearchOutput
    permissions = frozenset({"network"})
    timeout_seconds = 20.0
    retry = RetryPolicy(max_attempts=3, backoff_seconds=0.2)

    def __init__(self, provider: SearchProvider) -> None:
        self._provider = provider

    async def run(self, data: WebSearchInput, scope: ExecutionScope) -> WebSearchOutput:
        try:
            hits = await self._provider.search(data.query, data.max_results)
        except (ConnectionError, OSError) as exc:
            raise ToolTransientError(f"search provider unavailable: {exc}") from exc
        return WebSearchOutput(candidates=[
            SourceCandidate(source_id=source_id_for(h.url), url=h.url, title=h.title, publisher=h.publisher,
                            snippet=h.snippet, retrieved_via="web", metadata=h.metadata)
            for h in hits
        ])
