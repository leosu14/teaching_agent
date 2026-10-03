"""Deterministic mastery: how evidence moves a learner's concept estimates, and when a concept is reviewed next.

`MasteryUpdater` is the only code that changes a mastery estimate. It takes the previous state plus one piece of
LearningEvidence and returns the new state; the same inputs always give the same output. No model ever assigns a
mastery score: a model may grade an answer (interpret the evidence), the number is computed here.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Protocol

from app.schemas.learner import ConceptMastery, LearningEvidence, MasteryConfig, ReviewIntervals

# Thresholds of the learner snapshot (what the diagnostic and evaluation agents see).
PRIOR_MASTERY = MasteryConfig().prior_mastery
KNOWN = 0.7
MASTERED = 0.9
WEAK = 0.5
UNKNOWN = 0.3


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


class ReviewScheduler(Protocol):
    """Spaced-review policy: when a concept is next due. Replaceable (an SM-2 scheduler would implement this)."""

    def after_evidence(self, previous: ConceptMastery, updated: ConceptMastery, evidence: LearningEvidence
                       ) -> datetime: ...

    def after_exposure(self, concept: ConceptMastery, at: datetime) -> datetime: ...


class IntervalReviewScheduler:
    """A simple deterministic policy over mastery, correctness and recency.

    - Base interval by mastery: weak, medium or strong (`ReviewIntervals`).
    - Incorrect: review at the weak interval, whatever the mastery.
    - Correct after a gap at least as long as the base interval (it was retained): the interval is stretched by the
      retention multiplier, capped.
    """

    def __init__(self, intervals: ReviewIntervals | None = None) -> None:
        self.intervals = intervals or ReviewIntervals()

    def base_days(self, mastery: float) -> float:
        i = self.intervals
        if mastery < i.weak_below:
            return i.weak_days
        if mastery < i.strong_from:
            return i.medium_days
        return i.strong_days

    def after_evidence(self, previous: ConceptMastery, updated: ConceptMastery, evidence: LearningEvidence
                       ) -> datetime:
        i = self.intervals
        if evidence.correctness == "incorrect":
            return evidence.timestamp + timedelta(days=i.weak_days)
        days = self.base_days(updated.mastery)
        if evidence.correctness == "correct" and previous.last_assessed_at is not None:
            gap = (evidence.timestamp - previous.last_assessed_at).total_seconds() / 86400
            if gap >= days:
                days = min(i.max_days, max(days, gap) * i.retention_multiplier)
        return evidence.timestamp + timedelta(days=days)

    def after_exposure(self, concept: ConceptMastery, at: datetime) -> datetime:
        return at + timedelta(days=self.base_days(concept.mastery))


class MasteryUpdater:
    """previous ConceptMastery + LearningEvidence -> updated ConceptMastery (a new object; the input is untouched)."""

    def __init__(self, config: MasteryConfig | None = None, scheduler: ReviewScheduler | None = None) -> None:
        self.config = config or MasteryConfig()
        self.scheduler = scheduler or IntervalReviewScheduler(self.config.review)

    def initial(self, concept_id: str, name: str, subject: str) -> ConceptMastery:
        return ConceptMastery(concept_id=concept_id, name=name, subject=subject, mastery=self.config.prior_mastery)

    def rate(self, previous: ConceptMastery, evidence: LearningEvidence) -> float:
        c = self.config
        if evidence.source_type == "manual":  # a calibrated placement (e.g. entered by a teacher)
            return c.manual_weight
        success = evidence.score >= 0.5
        informativeness = evidence.difficulty if success else 1.0 - evidence.difficulty
        return (c.base_rate + c.informativeness_rate * informativeness) / (1 + c.rate_decay * previous.evidence_count)

    def apply(self, previous: ConceptMastery, evidence: LearningEvidence) -> ConceptMastery:
        if evidence.concept_id != previous.concept_id:
            raise ValueError(f"evidence for {evidence.concept_id} cannot update {previous.concept_id}")
        count = previous.evidence_count + 1
        incorrect = evidence.correctness == "incorrect"
        updated = previous.model_copy(update={
            "mastery": _clamp(previous.mastery + self.rate(previous, evidence) * (evidence.score - previous.mastery)),
            "confidence": _clamp(1 - self.config.confidence_base ** count),
            "evidence_count": count,
            "correct_count": previous.correct_count + (evidence.correctness == "correct"),
            "incorrect_count": previous.incorrect_count + incorrect,
            "incorrect_streak": previous.incorrect_streak + 1 if incorrect else 0,
            "last_assessed_at": evidence.timestamp,
            "last_updated_at": evidence.timestamp,
        })
        return updated.model_copy(update={"next_review_at": self.scheduler.after_evidence(previous, updated, evidence)})

    def expose(self, previous: ConceptMastery, at: datetime) -> ConceptMastery:
        """Being taught a concept is not evidence of mastery: it counts the exposure and schedules a review only."""
        return previous.model_copy(update={"exposures": previous.exposures + 1, "last_updated_at": at,
                                           "next_review_at": self.scheduler.after_exposure(previous, at)})

    def replay(self, seed: ConceptMastery, evidence: Iterable[LearningEvidence],
               exposures: Iterable[datetime] = ()) -> ConceptMastery:
        """Rebuild a concept's state from its full history (evidence and exposures in time order; evidence first on
        ties, then in recording order). The learner's current state is always reconstructible this way."""
        timeline = sorted([(e.timestamp, 0, i, e) for i, e in enumerate(evidence)]
                          + [(at, 1, i, None) for i, at in enumerate(exposures)], key=lambda x: x[:3])
        state = seed
        for at, _, _, item in timeline:
            state = self.apply(state, item) if item is not None else self.expose(state, at)
        return state
