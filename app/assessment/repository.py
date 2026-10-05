"""Assessment store. `app/storage/repositories.py` implements it on SQL; the in-memory version serves unit tests.

Items and rubrics are immutable (the same id with different content is refused). An attempt and its grade are stored
together, once: the attempt id is unique, and the attempt number is assigned in the same write (per learner and
item, in order). Nothing is ever overwritten, so every attempt stays auditable.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Protocol

from app.assessment.errors import AttemptExists, ItemConflict
from app.schemas.assessment import (
    AssessmentAttempt,
    AssessmentGrade,
    AssessmentItem,
    AssessmentRubric,
    AttemptOutcome,
)


class AssessmentRepository(Protocol):
    def save_item(self, item: AssessmentItem) -> bool:
        """True when stored now; False when the identical item exists; ItemConflict for different content."""
        ...

    def item(self, item_id: str) -> AssessmentItem | None: ...

    def save_rubric(self, rubric: AssessmentRubric) -> bool: ...

    def rubric(self, rubric_id: str) -> AssessmentRubric | None: ...

    def add_attempt(self, attempt: AssessmentAttempt, grade: AssessmentGrade) -> AssessmentAttempt:
        """Store both (the attempt number is assigned here); AttemptExists when the attempt id is taken."""
        ...

    def attempt(self, attempt_id: str) -> AssessmentAttempt | None: ...

    def attempts(self, item_id: str, learner_id: str) -> list[AssessmentAttempt]:
        """In attempt-number order."""
        ...

    def grade(self, grade_id: str) -> AssessmentGrade | None: ...

    def complete(self, attempt_id: str, outcome: AttemptOutcome | None, at: datetime) -> AssessmentAttempt:
        """Mark the attempt's publication done (once; a second call keeps the first outcome)."""
        ...


class InMemoryAssessmentRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, AssessmentItem] = {}
        self._rubrics: dict[str, AssessmentRubric] = {}
        self._attempts: dict[str, AssessmentAttempt] = {}
        self._grades: dict[str, AssessmentGrade] = {}

    @staticmethod
    def _save(table: dict, key: str, value) -> bool:
        current = table.get(key)
        if current is None:
            table[key] = value
            return True
        if current.content_hash() != value.content_hash():
            raise ItemConflict(f"{key} is already stored with different content")
        return False

    def save_item(self, item: AssessmentItem) -> bool:
        with self._lock:
            return self._save(self._items, item.assessment_item_id, item)

    def item(self, item_id: str) -> AssessmentItem | None:
        return self._items.get(item_id)

    def save_rubric(self, rubric: AssessmentRubric) -> bool:
        with self._lock:
            return self._save(self._rubrics, rubric.rubric_id, rubric)

    def rubric(self, rubric_id: str) -> AssessmentRubric | None:
        return self._rubrics.get(rubric_id)

    def add_attempt(self, attempt: AssessmentAttempt, grade: AssessmentGrade) -> AssessmentAttempt:
        with self._lock:
            if attempt.attempt_id in self._attempts or grade.grade_id in self._grades:
                raise AttemptExists(attempt.attempt_id)
            number = 1 + sum(1 for a in self._attempts.values() if a.assessment_item_id == attempt.assessment_item_id
                             and a.learner_id == attempt.learner_id)
            stored = attempt.model_copy(update={"attempt_number": number})
            self._attempts[attempt.attempt_id] = stored
            self._grades[grade.grade_id] = grade
            return stored.model_copy(deep=True)

    def attempt(self, attempt_id: str) -> AssessmentAttempt | None:
        found = self._attempts.get(attempt_id)
        return found.model_copy(deep=True) if found else None

    def attempts(self, item_id: str, learner_id: str) -> list[AssessmentAttempt]:
        return sorted((a.model_copy(deep=True) for a in self._attempts.values()
                       if a.assessment_item_id == item_id and a.learner_id == learner_id),
                      key=lambda a: a.attempt_number)

    def grade(self, grade_id: str) -> AssessmentGrade | None:
        return self._grades.get(grade_id)

    def complete(self, attempt_id: str, outcome: AttemptOutcome | None, at: datetime) -> AssessmentAttempt:
        with self._lock:
            current = self._attempts[attempt_id]
            if current.completed_at is None:
                current = current.model_copy(update={"completed_at": at, "outcome": outcome})
                self._attempts[attempt_id] = current
            return current.model_copy(deep=True)
