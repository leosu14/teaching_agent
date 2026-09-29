"""Document retrieval contract used for RAG (local knowledge base now, vector stores later)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from app.schemas.common import Schema


class Passage(Schema):
    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class Retriever(ABC):
    name: str

    @abstractmethod
    async def retrieve(self, query: str, k: int, filters: dict[str, str] | None = None) -> list[Passage]: ...
