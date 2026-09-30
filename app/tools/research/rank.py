"""Rank tool: deduplicates candidate sources and orders them with a pluggable Ranker."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.observability.scope import ExecutionScope
from app.providers.ranking.base import Ranker, RankingContext
from app.schemas.common import Schema, utcnow
from app.schemas.research import RankCandidate, RankingSignals, RankingWeights, SearchResult
from app.tools.base import Tool
from app.tools.research.dedup import DuplicateRecord, deduplicate


class RankInput(Schema):
    candidates: list[RankCandidate]
    language: str | None = None
    as_of: datetime | None = None  # reference time for freshness; defaults to now
    weights: RankingWeights = Field(default_factory=RankingWeights)


class RankedSource(Schema):
    rank: int = Field(ge=1)
    score: float
    signals: RankingSignals
    result: SearchResult  # with `source.reliability` filled in by the ranker
    matched_queries: list[str]


class RankOutput(Schema):
    ranked: list[RankedSource]
    duplicates: list[DuplicateRecord] = Field(default_factory=list)
    ranker: str


class RankSourcesTool(Tool[RankInput, RankOutput]):
    name = "research.rank"
    description = "Deduplicate candidate sources and rank them by relevance, quality, freshness, language and novelty."
    input_model = RankInput
    output_model = RankOutput

    def __init__(self, ranker: Ranker) -> None:
        self._ranker = ranker

    async def run(self, data: RankInput, scope: ExecutionScope) -> RankOutput:
        dedup = deduplicate(data.candidates)
        if not dedup.unique:
            return RankOutput(ranked=[], duplicates=dedup.duplicates, ranker=self._ranker.name)
        queries = list(dict.fromkeys(q for c in dedup.unique for q in c.matched_queries))
        context = RankingContext(queries=queries, language=data.language, as_of=data.as_of or utcnow())
        scores = {s.source_id: s for s in self._ranker.score(dedup.unique, context, data.weights)}
        order = sorted(range(len(dedup.unique)),
                       key=lambda i: (-scores[dedup.unique[i].result.source_id].score, i))
        ranked = []
        for position, i in enumerate(order, start=1):
            cand = dedup.unique[i]
            scored = scores[cand.result.source_id]
            source = cand.result.source.model_copy(update={"reliability": scored.reliability})
            ranked.append(RankedSource(rank=position, score=scored.score, signals=scored.signals,
                                       result=cand.result.model_copy(update={"source": source}),
                                       matched_queries=cand.matched_queries))
        return RankOutput(ranked=ranked, duplicates=dedup.duplicates, ranker=self._ranker.name)
