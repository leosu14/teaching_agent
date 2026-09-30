"""Ranking contract. A ranker scores deduplicated candidates; later adapters (a rerank API, a learned
model) implement the same interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import Field

from app.schemas.common import Schema
from app.schemas.research import RankCandidate, RankingSignals, RankingWeights, SourceReliability


class RankingContext(Schema):
    queries: list[str] = Field(min_length=1)  # the query texts that found the candidates
    language: str | None = None
    as_of: datetime


class Scored(Schema):
    source_id: str
    score: float
    signals: RankingSignals
    reliability: SourceReliability


class Ranker(ABC):
    name: str

    @abstractmethod
    def score(self, candidates: list[RankCandidate], context: RankingContext,
              weights: RankingWeights) -> list[Scored]:
        """Score every candidate. Order of the returned list does not matter; the caller sorts."""
