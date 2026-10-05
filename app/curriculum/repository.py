"""Curriculum stores. Versions are append-only and immutable; the record is the current-version pointer; the progress
snapshot is the last progress seen (to detect transitions once). `app/storage/repositories.py` implements the
protocol on SQLite; the in-memory version serves tests and single-process use. The learning-cycle store follows the
same pattern."""

from __future__ import annotations

from typing import Protocol

from app.schemas.curriculum import CurriculumProgress, CurriculumRecord, CurriculumVersion, VersionConflict
from app.schemas.learning_cycle import TERMINAL_CYCLE_STATUSES, CycleConflict, CycleRequestRecord, LearningCycle

__all__ = ["CurriculumRepository", "InMemoryCurriculumRepository", "InMemoryLearningCycleRepository",
           "LearningCycleRepository", "VersionConflict", "same_version"]


class CurriculumRepository(Protocol):
    def save_record(self, record: CurriculumRecord) -> None: ...

    def record(self, curriculum_id: str) -> CurriculumRecord | None: ...

    def records_for_learner(self, learner_id: str) -> list[CurriculumRecord]: ...

    def add_version(self, version: CurriculumVersion) -> bool:
        """Store a new version and return True; False for an identical re-add; VersionConflict otherwise."""
        ...

    def version(self, version_id: str) -> CurriculumVersion | None: ...

    def versions(self, curriculum_id: str) -> list[CurriculumVersion]:
        """Oldest first."""
        ...

    def save_progress(self, progress: CurriculumProgress) -> None: ...

    def progress(self, curriculum_id: str) -> CurriculumProgress | None: ...


def same_version(a: CurriculumVersion, b: CurriculumVersion) -> bool:
    """Identical apart from the creation time (a resumed task re-adding the version it already stored)."""
    return a.model_dump(exclude={"created_at"}) == b.model_dump(exclude={"created_at"})


class InMemoryCurriculumRepository:
    def __init__(self) -> None:
        self._records: dict[str, CurriculumRecord] = {}
        self._versions: dict[str, CurriculumVersion] = {}
        self._progress: dict[str, CurriculumProgress] = {}

    def save_record(self, record: CurriculumRecord) -> None:
        self._records[record.curriculum_id] = record.model_copy(deep=True)

    def record(self, curriculum_id: str) -> CurriculumRecord | None:
        found = self._records.get(curriculum_id)
        return found.model_copy(deep=True) if found else None

    def records_for_learner(self, learner_id: str) -> list[CurriculumRecord]:
        return sorted((r.model_copy(deep=True) for r in self._records.values() if r.learner_id == learner_id),
                      key=lambda r: r.curriculum_id)

    def add_version(self, version: CurriculumVersion) -> bool:
        existing = self._versions.get(version.version_id)
        if existing is not None:
            if not same_version(existing, version):
                raise VersionConflict(f"curriculum version {version.version_id} exists with different content")
            return False
        self._versions[version.version_id] = version.model_copy(deep=True)
        return True

    def version(self, version_id: str) -> CurriculumVersion | None:
        found = self._versions.get(version_id)
        return found.model_copy(deep=True) if found else None

    def versions(self, curriculum_id: str) -> list[CurriculumVersion]:
        return sorted((v.model_copy(deep=True) for v in self._versions.values() if v.curriculum_id == curriculum_id),
                      key=lambda v: v.version)

    def save_progress(self, progress: CurriculumProgress) -> None:
        self._progress[progress.curriculum_id] = progress.model_copy(deep=True)

    def progress(self, curriculum_id: str) -> CurriculumProgress | None:
        found = self._progress.get(curriculum_id)
        return found.model_copy(deep=True) if found else None


class LearningCycleRepository(Protocol):
    """Learning cycles, the learner's active-cycle slot and the learner responses they received. `create` and `apply`
    are each one transaction; `apply` only succeeds against the version the change was computed from."""

    def create(self, cycle: LearningCycle) -> bool:
        """Store a new cycle and take the learner's active slot. False if the cycle id exists (the same key);
        CycleConflict if another cycle holds the slot."""
        ...

    def apply(self, cycle: LearningCycle, expected_version: int, request: CycleRequestRecord | None = None) -> None:
        """Write the changed cycle (releasing the slot when it ends); CycleConflict on a stale version."""
        ...

    def get(self, cycle_id: str) -> LearningCycle | None: ...

    def active_for(self, learner_id: str) -> LearningCycle | None: ...

    def for_learner(self, learner_id: str) -> list[LearningCycle]:
        """Newest first."""
        ...

    def request(self, cycle_id: str, client_response_id: str) -> CycleRequestRecord | None: ...

    def mark_applied(self, cycle_id: str, client_response_id: str) -> None: ...

    def forget_request(self, cycle_id: str, client_response_id: str) -> None:
        """A response the child refused (invalid input): the id may be used again."""
        ...


class InMemoryLearningCycleRepository:
    def __init__(self) -> None:
        self._cycles: dict[str, LearningCycle] = {}
        self._slots: dict[str, str] = {}
        self._requests: dict[tuple[str, str], CycleRequestRecord] = {}

    def create(self, cycle: LearningCycle) -> bool:
        if cycle.cycle_id in self._cycles:
            return False
        holder = self._slots.get(cycle.learner_id)
        if holder is not None:
            raise CycleConflict(f"learner has an active learning cycle {holder}")
        self._cycles[cycle.cycle_id] = cycle.model_copy(deep=True)
        self._slots[cycle.learner_id] = cycle.cycle_id
        return True

    def apply(self, cycle: LearningCycle, expected_version: int, request: CycleRequestRecord | None = None) -> None:
        stored = self._cycles.get(cycle.cycle_id)
        if stored is None or stored.version != expected_version:
            raise CycleConflict(f"cycle {cycle.cycle_id} changed concurrently (expected version {expected_version})")
        if request is not None:
            key = (request.cycle_id, request.client_response_id)
            if key in self._requests:
                raise CycleConflict(f"response {request.client_response_id} was already received")
            self._requests[key] = request.model_copy(deep=True)
        self._cycles[cycle.cycle_id] = cycle.model_copy(deep=True)
        if cycle.status in TERMINAL_CYCLE_STATUSES and self._slots.get(cycle.learner_id) == cycle.cycle_id:
            del self._slots[cycle.learner_id]

    def get(self, cycle_id: str) -> LearningCycle | None:
        found = self._cycles.get(cycle_id)
        return found.model_copy(deep=True) if found else None

    def active_for(self, learner_id: str) -> LearningCycle | None:
        cycle_id = self._slots.get(learner_id)
        return self.get(cycle_id) if cycle_id else None

    def for_learner(self, learner_id: str) -> list[LearningCycle]:
        return sorted((c.model_copy(deep=True) for c in self._cycles.values() if c.learner_id == learner_id),
                      key=lambda c: (c.created_at, c.cycle_id), reverse=True)

    def request(self, cycle_id: str, client_response_id: str) -> CycleRequestRecord | None:
        found = self._requests.get((cycle_id, client_response_id))
        return found.model_copy(deep=True) if found else None

    def mark_applied(self, cycle_id: str, client_response_id: str) -> None:
        found = self._requests.get((cycle_id, client_response_id))
        if found is not None:
            found.applied = True

    def forget_request(self, cycle_id: str, client_response_id: str) -> None:
        self._requests.pop((cycle_id, client_response_id), None)
