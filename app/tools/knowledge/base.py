"""The existing retrieval knowledge base (fixtures/demo/knowledge_base.json and friends) behind the structured
KnowledgeBase interface, plus the `knowledge.concepts` tool."""

from __future__ import annotations

from pydantic import Field

from app.observability.scope import ExecutionScope
from app.pedagogy.graph import ConceptGraph, ConceptGraphError
from app.pedagogy.knowledge import KnowledgeBase
from app.providers.retrieval.base import Retriever
from app.schemas.common import Schema
from app.schemas.concepts import Concept
from app.schemas.pedagogy import ConceptSet
from app.tools.base import Tool, ToolError

MAX_CONCEPTS = 1000


class RetrieverKnowledgeBase(KnowledgeBase):
    """Concept documents (`metadata.kind == "concept"`) of a Retriever: the subject is the domain, the document's
    concept metadata carries id, name, level, prerequisites and description."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def concepts(self, domain: str, topic: str | None = None, level: str | None = None) -> list[Concept]:
        filters = {"kind": "concept", "subject": domain}
        if topic:
            filters["topic"] = topic
        passages = await self._retriever.retrieve(" ".join(filter(None, [topic, level])) or domain, MAX_CONCEPTS,
                                                  filters)
        concepts = []
        for p in sorted(passages, key=lambda p: p.doc_id):
            raw = p.metadata.get("concept")
            if not raw:
                raise ToolError(f"knowledge-base concept document {p.doc_id} has no concept metadata")
            concept = Concept.model_validate({**raw, "domain": domain, "topic": p.metadata.get("topic")})
            if level is None or concept.level == level:
                concepts.append(concept)
        return concepts


class ConceptQuery(Schema):
    domain: str = Field(min_length=1)
    topic: str | None = None


class KnowledgeConceptsTool(Tool[ConceptQuery, ConceptSet]):
    name = "knowledge.concepts"
    description = "The domain's concepts and prerequisite graph from the knowledge base (validated, prerequisites first)."
    input_model = ConceptQuery
    output_model = ConceptSet
    permissions = frozenset({"knowledge:read"})

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self._knowledge = knowledge

    async def run(self, data: ConceptQuery, scope: ExecutionScope) -> ConceptSet:
        try:
            graph = await self._knowledge.graph(data.domain)
        except ConceptGraphError as exc:
            raise ToolError(f"invalid concept graph for {data.domain}: {exc}") from exc
        if data.topic:
            topic = {c.concept_id for c in await self._knowledge.concepts(data.domain, data.topic)}
            graph = ConceptGraph(graph.concept(c) for c in graph.closure(topic))
        return ConceptSet(domain=data.domain, concepts=graph.concepts())
