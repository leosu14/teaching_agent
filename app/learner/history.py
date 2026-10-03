"""Learner history stores: evidence, learning events and goals. Append-only where history is concerned.

The protocols are what the learner layer needs; `app/storage/repositories.py` implements them on SQLite and the
in-memory versions here serve tests and single-process use.
"""

from __future__ import annotations

from typing import Protocol

from app.schemas.learner import EvidenceConflict, LearningEvent, LearningEvidence, LearningGoal

__all__ = ["EvidenceConflict", "EvidenceRepository", "GoalRepository", "LearningEventRepository",
           "InMemoryEvidenceRepository", "InMemoryGoalRepository", "InMemoryLearningEventRepository"]


class EvidenceRepository(Protocol):
    def add(self, evidence: LearningEvidence) -> bool:
        """Store new evidence and return True; return False for an identical re-record; raise EvidenceConflict when
        the id exists with different content."""
        ...

    def for_learner(self, learner_id: str, concept_id: str | None = None) -> list[LearningEvidence]:
        """In recording order."""
        ...


class LearningEventRepository(Protocol):
    def add(self, event: LearningEvent) -> bool: ...

    def for_learner(self, learner_id: str) -> list[LearningEvent]: ...


class GoalRepository(Protocol):
    def save(self, goal: LearningGoal) -> None: ...

    def get(self, goal_id: str) -> LearningGoal | None: ...

    def for_learner(self, learner_id: str) -> list[LearningGoal]: ...


class InMemoryEvidenceRepository:
    def __init__(self) -> None:
        self._rows: dict[str, LearningEvidence] = {}

    def add(self, evidence: LearningEvidence) -> bool:
        existing = self._rows.get(evidence.evidence_id)
        if existing is not None:
            if existing != evidence:
                raise EvidenceConflict(f"evidence {evidence.evidence_id} is already recorded with different content")
            return False
        self._rows[evidence.evidence_id] = evidence
        return True

    def for_learner(self, learner_id: str, concept_id: str | None = None) -> list[LearningEvidence]:
        return [e for e in self._rows.values()
                if e.learner_id == learner_id and (concept_id is None or e.concept_id == concept_id)]


class InMemoryLearningEventRepository:
    def __init__(self) -> None:
        self._rows: dict[str, LearningEvent] = {}

    def add(self, event: LearningEvent) -> bool:
        if event.event_id in self._rows:
            return False
        self._rows[event.event_id] = event
        return True

    def for_learner(self, learner_id: str) -> list[LearningEvent]:
        return [e for e in self._rows.values() if e.learner_id == learner_id]


class InMemoryGoalRepository:
    def __init__(self) -> None:
        self._rows: dict[str, LearningGoal] = {}

    def save(self, goal: LearningGoal) -> None:
        self._rows[goal.goal_id] = goal.model_copy(deep=True)

    def get(self, goal_id: str) -> LearningGoal | None:
        goal = self._rows.get(goal_id)
        return goal.model_copy(deep=True) if goal else None

    def for_learner(self, learner_id: str) -> list[LearningGoal]:
        return [g.model_copy(deep=True) for g in self._rows.values() if g.learner_id == learner_id]
