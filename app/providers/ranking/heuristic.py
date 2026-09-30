"""Deterministic ranker: lexical relevance, source-type quality, age, language match and publisher novelty."""

from __future__ import annotations

import re
from collections import Counter

from app.providers.ranking.base import Ranker, RankingContext, Scored
from app.schemas.research import RankCandidate, RankingSignals, RankingWeights, SourceReliability

STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it of on or the this to with what why when which who".split()
)
# Quality of a source by the type its provider reported. Unknown types get no score (neutral in ranking).
SOURCE_TYPE_QUALITY = {
    "reference": 0.95, "academic": 0.95, "knowledge_base": 0.9, "educational": 0.85, "official": 0.85,
    "news": 0.7, "blog": 0.45, "forum": 0.3, "social": 0.2,
}
NEUTRAL = 0.5
FRESHNESS_HALF_LIFE_YEARS = 5.0


def terms(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and len(t) > 1}


class HeuristicRanker(Ranker):
    name = "heuristic"

    def __init__(self, source_type_quality: dict[str, float] | None = None) -> None:
        self._quality = dict(SOURCE_TYPE_QUALITY if source_type_quality is None else source_type_quality)

    def score(self, candidates: list[RankCandidate], context: RankingContext,
              weights: RankingWeights) -> list[Scored]:
        total = weights.relevance + weights.quality + weights.freshness + weights.language + weights.novelty
        if total <= 0:
            raise ValueError("ranking weights must not all be zero")
        # Novelty: later candidates from an already-seen publisher are worth less (in input order).
        seen: Counter[str] = Counter()
        scored = []
        for cand in candidates:
            src = cand.result.source
            notes: list[str] = []
            relevance = self._relevance(cand)
            quality, reliability = self._quality_of(src.source_type, notes)
            freshness = self._freshness(src.published_at, context, notes)
            language = self._language(src.language, context.language, notes)
            publisher = (src.publisher or src.canonical_url).lower()
            novelty = 1.0 / (1 + seen[publisher])
            seen[publisher] += 1
            signals = RankingSignals(relevance=relevance, quality=quality, freshness=freshness, language=language,
                                     novelty=round(novelty, 4), notes=notes)
            score = (weights.relevance * relevance + weights.quality * quality + weights.freshness * freshness
                     + weights.language * language + weights.novelty * novelty) / total
            scored.append(Scored(source_id=src.source_id, score=round(score, 6), signals=signals,
                                 reliability=reliability))
        return scored

    def _relevance(self, cand: RankCandidate) -> float:
        r = cand.result
        doc = terms(f"{r.title} {r.snippet} {r.content or ''}")
        best = 0.0
        for query in cand.matched_queries:
            wanted = terms(query)
            if wanted:
                best = max(best, len(wanted & doc) / len(wanted))
        return round(best, 4)

    def _quality_of(self, source_type: str, notes: list[str]) -> tuple[float, SourceReliability]:
        score = self._quality.get(source_type)
        if score is None:
            notes.append(f"source type '{source_type}' has no quality rating")
            return NEUTRAL, SourceReliability(score=None, basis=f"source_type={source_type} (unrated)",
                                              assessed_by=f"ranker:{self.name}")
        return score, SourceReliability(score=score, basis=f"source_type={source_type}",
                                        assessed_by=f"ranker:{self.name}")

    def _freshness(self, published, context: RankingContext, notes: list[str]) -> float:
        if published is None:
            notes.append("no publication date")
            return NEUTRAL
        years = max(0.0, (context.as_of.date() - published).days / 365.25)
        return round(0.5 ** (years / FRESHNESS_HALF_LIFE_YEARS), 4)

    def _language(self, source_language: str | None, wanted: str | None, notes: list[str]) -> float:
        if wanted is None or source_language is None:
            notes.append("language unknown")
            return NEUTRAL
        return 1.0 if source_language.split("-")[0].lower() == wanted.split("-")[0].lower() else 0.0
