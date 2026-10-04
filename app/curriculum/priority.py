"""The next-action priority score: explicit, simple and deterministic.

    score = prerequisite_ready x (sum_k weight_k x factor_k) / (sum_k weight_k)

Every factor is in [0, 1]:

    deficit            max(0, target_mastery - mastery) / target_mastery        (0 for a mastered objective)
    objective_priority (5 - objective.priority) / 4                              (priority 1 -> 1.0, 5 -> 0.0)
    goal_priority      (5 - goal.priority) / 4
    review_urgency     0 unless the review is due; then 0.5 + 0.5 x min(1, overdue_days / review_interval_days)
    deadline_pressure  0 without a target date; else 1 - min(1, days_left / deadline_horizon_days); 1 once passed
    recent_failure     min(1, incorrect_streak / repeated_failure_streak)
    recency            min(1, days_since_last_evidence / recency_horizon_days); 1 when never assessed

`prerequisite_ready` is 1 when every prerequisite of the objective is at or above the prerequisite threshold, else
0: an objective whose prerequisites are not ready is never selected, whatever its other factors. Weights are in
`CurriculumConfig.weights`. Time enters only through the explicit `as_of`, never the wall clock.
"""

from __future__ import annotations

from datetime import datetime

from app.schemas.curriculum import (
    CurriculumConfig,
    CurriculumObjective,
    ObjectiveProgress,
    PriorityFactors,
    PriorityScore,
)
from app.schemas.learner import ConceptMastery


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def _days(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 86400


def priority_factors(objective: CurriculumObjective, progress: ObjectiveProgress, state: ConceptMastery | None,
                     goal_priority: int, target_date: datetime | None, as_of: datetime,
                     config: CurriculumConfig, *, reviewing: bool = False) -> PriorityFactors:
    mastery = progress.current_mastery
    deficit = 0.0 if reviewing else max(0.0, objective.target_mastery - mastery) / objective.target_mastery
    urgency = 0.0
    review = progress.review
    if review is not None and review.due and review.next_review_at is not None:
        interval = review.review_interval_days or 1.0
        urgency = 0.5 + 0.5 * min(1.0, _days(as_of, review.next_review_at) / max(interval, 1e-9))
    pressure = 0.0
    if target_date is not None:
        left = _days(target_date, as_of)
        pressure = 1.0 if left <= 0 else 1.0 - min(1.0, left / config.deadline_horizon_days)
    last = state.last_assessed_at if state is not None and state.evidence_count else None
    recency = 1.0 if last is None else min(1.0, max(0.0, _days(as_of, last)) / config.recency_horizon_days)
    return PriorityFactors(
        deficit=_clamp(deficit),
        objective_priority=_clamp((5 - objective.priority) / 4),
        goal_priority=_clamp((5 - goal_priority) / 4),
        review_urgency=_clamp(urgency),
        deadline_pressure=_clamp(pressure),
        recent_failure=_clamp(progress.incorrect_streak / config.repeated_failure_streak),
        recency=_clamp(recency),
    )


def priority_score(factors: PriorityFactors, prerequisite_ready: bool, config: CurriculumConfig) -> PriorityScore:
    weights = config.weights.model_dump()
    values = factors.model_dump()
    weighted = sum(weights[k] * values[k] for k in weights) / sum(weights.values())
    return PriorityScore(factors=factors, prerequisite_ready=prerequisite_ready,
                         score=_clamp(weighted if prerequisite_ready else 0.0))
