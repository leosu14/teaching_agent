"""Curriculum stores. Versions are append-only and immutable; the record is the current-version pointer; the progress
snapshot is the last progress seen (to detect transitions once). `app/storage/repositories.py` implements the
protocol on SQLite; the in-memory version serves tests and single-process use."""

from __future__ import annotations

from typing import Protocol

from app.schemas.curriculum import CurriculumProgress, CurriculumRecord, CurriculumVersion, VersionConflict

__all__ = ["CurriculumRepository", "InMemoryCurriculumRepository", "VersionConflict", "same_version"]


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
