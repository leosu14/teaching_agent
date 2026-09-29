"""Mastery update rules: how evidence and exposure move a learner's concept estimates."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.schemas.learner import ConceptMastery

PRIOR_MASTERY = 0.3
KNOWN = 0.7
MASTERED = 0.9
WEAK = 0.5
UNKNOWN = 0.3


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def review_interval(mastery: float) -> timedelta:
    if mastery < WEAK:
        return timedelta(days=1)
    if mastery < 0.8:
        return timedelta(days=3)
    return timedelta(days=7)


def apply_evidence(concept: ConceptMastery, *, correct: bool, difficulty: float, now: datetime) -> None:
    """Move mastery towards 1 (correct) or 0 (wrong). Hard successes and easy failures carry more weight;
    the step shrinks as evidence accumulates so estimates stabilise."""
    target = 1.0 if correct else 0.0
    informativeness = difficulty if correct else 1.0 - difficulty
    rate = (0.35 + 0.3 * informativeness) / (1 + 0.15 * concept.evidence_count)
    concept.mastery = _clamp(concept.mastery + rate * (target - concept.mastery))
    concept.evidence_count += 1
    concept.confidence = _clamp(1 - 0.6 ** concept.evidence_count)
    concept.last_evidence_at = now
    concept.next_review_at = now + review_interval(concept.mastery)


def apply_exposure(concept: ConceptMastery, *, now: datetime) -> None:
    """Being taught a concept raises mastery a little and schedules a review; it is not proof of mastery."""
    concept.exposures += 1
    concept.mastery = _clamp(concept.mastery + 0.1 * (1 - concept.mastery))
    concept.next_review_at = now + review_interval(concept.mastery)
