"""Deterministic outcome classification. The model never chooses the outcome.

  score >= correct_threshold  -> CORRECT  (capped at PARTIAL when a required criterion is unmet)
  score >= partial_threshold  -> PARTIAL
  otherwise                   -> INCORRECT

and, for a semantic grade, its confidence:

  confidence >= accept_confidence                    -> the outcome above stands
  min_confidence <= confidence < accept_confidence   -> mid_confidence "partial": CORRECT is capped at PARTIAL and
                                                        INCORRECT becomes UNCERTAIN; "uncertain": UNCERTAIN
  confidence < min_confidence                        -> UNCERTAIN

A capped grade keeps the highest scale point below the correct threshold, so its score agrees with its outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.assessment import AssessmentConfig, AssessmentOutcome, ScoreScale

Outcome = AssessmentOutcome


@dataclass(frozen=True)
class Classification:
    outcome: AssessmentOutcome
    fraction: float
    reason: str | None = None  # why the grade is UNCERTAIN, or why it was capped


def by_score(fraction: float, correct_threshold: float, partial_threshold: float) -> AssessmentOutcome:
    if fraction >= correct_threshold:
        return Outcome.CORRECT
    if fraction >= partial_threshold:
        return Outcome.PARTIAL
    return Outcome.INCORRECT


def cap(fraction: float, correct_threshold: float, partial_threshold: float, scale: ScoreScale) -> float:
    below = [p for p in scale.points if p < correct_threshold]
    capped = min(fraction, max(below) if below else partial_threshold)
    return round(max(capped, partial_threshold), 4)


def classify(fraction: float, confidence: float, *, correct_threshold: float, config: AssessmentConfig,
             scale: ScoreScale, required_unmet: bool = False) -> Classification:
    partial = config.partial_threshold
    if confidence < config.min_confidence:
        return Classification(Outcome.UNCERTAIN, fraction,
                              f"confidence {confidence:.2f} is below the minimum {config.min_confidence:.2f}")
    outcome = by_score(fraction, correct_threshold, partial)
    reason = None
    if outcome == Outcome.CORRECT and required_unmet:
        outcome, fraction = Outcome.PARTIAL, cap(fraction, correct_threshold, partial, scale)
        reason = "a required criterion is not met"
    if confidence < config.accept_confidence:
        band = f"confidence {confidence:.2f} is below {config.accept_confidence:.2f}"
        if config.mid_confidence == "uncertain" or outcome == Outcome.INCORRECT:
            return Classification(Outcome.UNCERTAIN, fraction, band)
        if outcome == Outcome.CORRECT:
            return Classification(Outcome.PARTIAL, cap(fraction, correct_threshold, partial, scale),
                                  f"{band}: capped at PARTIAL")
    return Classification(outcome, fraction, reason)
