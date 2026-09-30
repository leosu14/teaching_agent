"""RAG tools over the knowledge base: passage retrieval and concept-map lookup."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import Field

from app.observability.scope import ExecutionScope
from app.providers.retrieval.base import Passage, Retriever
from app.schemas.common import Schema, utcnow
from app.schemas.lesson import ConceptEntry, ConceptRef, ProbeSpec
from app.schemas.research import SearchResult, Source
from app.tools.base import Tool, ToolError, ToolTransientError
from app.utils.urls import canonical_url, source_id_for


class RetrieveInput(Schema):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)


class RetrieveOutput(Schema):
    results: list[SearchResult]


def kb_url(doc_id: str) -> str:
    return f"kb://{doc_id}"


class RetrievalTool(Tool[RetrieveInput, RetrieveOutput]):
    name = "rag.retrieve"
    description = "Retrieve knowledge-base passages as traceable source candidates."
    input_model = RetrieveInput
    output_model = RetrieveOutput
    permissions = frozenset({"knowledge:read"})

    def __init__(self, retriever: Retriever, clock: Callable[[], datetime] = utcnow) -> None:
        self._retriever = retriever
        self._clock = clock

    async def run(self, data: RetrieveInput, scope: ExecutionScope) -> RetrieveOutput:
        try:
            passages = await self._retriever.retrieve(data.query, data.k, data.filters or None)
        except (ConnectionError, OSError) as exc:
            raise ToolTransientError(f"retriever '{self._retriever.name}' unavailable: {exc}") from exc
        retrieved_at = self._clock()
        scope.usage.record_service(service=f"retrieval:{self._retriever.name}", results=len(passages))
        return RetrieveOutput(results=[self._result(p, rank, retrieved_at) for rank, p in enumerate(passages, 1)])

    def _result(self, p: Passage, rank: int, retrieved_at: datetime) -> SearchResult:
        url = kb_url(p.doc_id)
        meta = p.metadata
        source = Source(
            source_id=source_id_for(url), url=url, canonical_url=canonical_url(url), title=p.title,
            publisher=meta.get("publisher"), author=meta.get("author"), published_at=meta.get("published_at"),
            retrieved_at=retrieved_at, language=meta.get("language"), source_type="knowledge_base",
            retrieved_via="knowledge_base", provider=self._retriever.name,
            metadata={k: v for k, v in meta.items()
                      if k not in {"publisher", "author", "published_at", "language"}},
        )
        return SearchResult(source_id=source.source_id, title=p.title, url=url, snippet=p.text[:300],
                            content=p.text, rank=rank, provider_score=p.score, source=source)


class ConceptMapInput(Schema):
    subject: str
    topic: str
    level: str | None = None
    k: int = Field(default=12, ge=1, le=100)


class ConceptMapOutput(Schema):
    concepts: list[ConceptEntry]


class ConceptMapTool(Tool[ConceptMapInput, ConceptMapOutput]):
    name = "rag.concept_map"
    description = "Look up the concepts (with prerequisites and probe hints) the knowledge base holds for a topic."
    input_model = ConceptMapInput
    output_model = ConceptMapOutput
    permissions = frozenset({"knowledge:read"})

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def run(self, data: ConceptMapInput, scope: ExecutionScope) -> ConceptMapOutput:
        filters = {"kind": "concept", "subject": data.subject, "topic": data.topic}
        query = " ".join(filter(None, [data.topic, data.level]))
        passages = await self._retriever.retrieve(query, data.k, filters)
        entries = []
        for p in passages:
            raw = p.metadata.get("concept")
            if not raw:
                raise ToolError(f"knowledge-base concept document {p.doc_id} has no concept metadata")
            entries.append(ConceptEntry(
                concept=ConceptRef.model_validate(raw),
                probes=[ProbeSpec.model_validate(pr) for pr in p.metadata.get("probes", [])],
                source_id=source_id_for(kb_url(p.doc_id)),
            ))
        return ConceptMapOutput(concepts=_prerequisite_order(entries))


def _prerequisite_order(entries: list[ConceptEntry]) -> list[ConceptEntry]:
    """Stable topological order so prerequisites come before the concepts that need them."""
    by_id = {e.concept.concept_id: e for e in entries}
    ordered: list[ConceptEntry] = []
    done: set[str] = set()

    def visit(cid: str, stack: frozenset[str]) -> None:
        if cid in done or cid not in by_id or cid in stack:
            return
        for pre in by_id[cid].concept.prerequisites:
            visit(pre, stack | {cid})
        done.add(cid)
        ordered.append(by_id[cid])

    for entry in entries:
        visit(entry.concept.concept_id, frozenset())
    return ordered
