"""RAG tools over the knowledge base: passage retrieval and concept-map lookup."""

from __future__ import annotations

from pydantic import Field

from app.observability.scope import ExecutionScope
from app.providers.retrieval.base import Retriever
from app.schemas.common import Schema
from app.schemas.lesson import ConceptEntry, ConceptRef, ProbeSpec, SourceCandidate
from app.tools.base import Tool, ToolError


class RetrieveInput(Schema):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)


class RetrieveOutput(Schema):
    passages: list[SourceCandidate]


class RetrievalTool(Tool[RetrieveInput, RetrieveOutput]):
    name = "rag.retrieve"
    description = "Retrieve knowledge-base passages as traceable source candidates."
    input_model = RetrieveInput
    output_model = RetrieveOutput
    permissions = frozenset({"knowledge:read"})

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def run(self, data: RetrieveInput, scope: ExecutionScope) -> RetrieveOutput:
        passages = await self._retriever.retrieve(data.query, data.k, data.filters or None)
        return RetrieveOutput(passages=[
            SourceCandidate(
                source_id=f"kb_{p.doc_id}", url=f"kb://{p.doc_id}", title=p.title,
                publisher=str(p.metadata.get("publisher", "knowledge base")), snippet=p.text,
                retrieved_via="knowledge_base", metadata=p.metadata,
            )
            for p in passages
        ])


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
                source_id=f"kb_{p.doc_id}",
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
