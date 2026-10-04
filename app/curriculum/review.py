"""Review scheduling for curriculum objectives: a mastered objective does not disappear, it comes back for review.

The schedule itself is the learner layer's: `MasteryUpdater` sets `next_review_at` on every piece of evidence and
every exposure through its replaceable `ReviewScheduler` (app/learner/mastery.py, `IntervalReviewScheduler`: weak /
medium / strong intervals, sooner after an error, longer after a retained success). A `ReviewPolicy` turns that state
into the curriculum's view of it; replacing it (e.g. with a spaced-repetition model) changes nothing else here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.schemas.curriculum import ReviewSchedule
from app.schemas.learner import ConceptMastery


class ReviewPolicy(Protocol):
    def schedule(self, concept_id: str, state: ConceptMastery | None, review_count: int,
                 as_of: datetime) -> ReviewSchedule: ...


class MasteryReviewPolicy:
    """Reads the learner model's review date: due when `next_review_at <= as_of`; the interval is the time between the
    last evidence and the next review; `review_count` counts the concept's recorded reviews."""

    def schedule(self, concept_id: str, state: ConceptMastery | None, review_count: int,
                 as_of: datetime) -> ReviewSchedule:
        if state is None:
            return ReviewSchedule(concept_id=concept_id, review_count=review_count)
        last = state.last_assessed_at
        nxt = state.next_review_at
        interval = round((nxt - last).total_seconds() / 86400, 4) if nxt is not None and last is not None else None
        return ReviewSchedule(concept_id=concept_id, next_review_at=nxt, last_reviewed_at=last,
                              review_interval_days=interval, review_count=review_count,
                              due=nxt is not None and nxt <= as_of)
