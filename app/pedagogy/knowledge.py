"""KnowledgeBase: the structured source of concepts, prerequisites, levels and relationships for a domain.

Subject-specific knowledge lives behind this interface (a JSON fixture, a curriculum database, ...), never in the
engine. `app/tools/knowledge/` adapts the existing retrieval knowledge base to it."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.pedagogy.graph import ConceptGraph
from app.schemas.concepts import Concept


class KnowledgeBase(ABC):
    @abstractmethod
    async def concepts(self, domain: str, topic: str | None = None, level: str | None = None) -> list[Concept]:
        """The domain's concepts, optionally narrowed to a topic and/or level."""

    async def concept(self, domain: str, concept_id: str) -> Concept:
        return (await self.graph(domain)).concept(concept_id)

    async def graph(self, domain: str) -> ConceptGraph:
        """The domain's full prerequisite graph (validated: known prerequisites, no cycles)."""
        return ConceptGraph(await self.concepts(domain))

    async def prerequisites(self, domain: str, concept_id: str) -> list[str]:
        return (await self.graph(domain)).prerequisites(concept_id)

    async def levels(self, domain: str) -> list[str]:
        return sorted({c.level for c in await self.concepts(domain) if c.level})

    async def relationships(self, domain: str) -> list[tuple[str, str]]:
        """(prerequisite, dependent) edges."""
        return [(pre, c.concept_id) for c in (await self.graph(domain)).concepts() for pre in c.prerequisites]
