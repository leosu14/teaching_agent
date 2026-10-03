"""Concepts: what a knowledge base teaches. Subject-independent: a concept is data, never code."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import Schema


class ConceptRef(Schema):
    """A lesson's reference to a concept (what agents and lesson schemas carry)."""

    concept_id: str
    name: str
    level: str | None = None
    prerequisites: list[str] = Field(default_factory=list)
    description: str = ""


class Concept(ConceptRef):
    """A knowledge-base concept: a node of the concept graph within one domain (a subject such as spanish,
    mathematics or history). `prerequisites` are the edges."""

    domain: str = Field(min_length=1)
    topic: str | None = None

    def ref(self) -> ConceptRef:
        return ConceptRef.model_validate(self.model_dump(exclude={"domain", "topic"}))
