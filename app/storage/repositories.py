"""SQL repositories: persistence only, no business rules."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.schemas.artifact import Artifact
from app.schemas.curriculum import CurriculumProgress, CurriculumRecord, CurriculumVersion, VersionConflict
from app.schemas.events import Event
from app.schemas.learner import EvidenceConflict, LearnerProfile, LearningEvent, LearningEvidence, LearningGoal
from app.schemas.task import Task
from app.storage.orm import (
    ArtifactRow,
    CurriculumProgressRow,
    CurriculumRow,
    CurriculumVersionRow,
    EventRow,
    LearnerRow,
    LearningEventRow,
    LearningEvidenceRow,
    LearningGoalRow,
    TaskRow,
)


class NotFound(KeyError):
    pass


class SqlTaskRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def save(self, task: Task) -> None:
        with self._sessions.begin() as s:
            s.merge(TaskRow(
                task_id=task.task_id, user_id=task.user_id, learner_id=task.learner_id,
                status=task.status.value, created_at=task.created_at, updated_at=task.updated_at,
                body=task.model_dump_json(),
            ))

    def get(self, task_id: str) -> Task:
        with self._sessions() as s:
            row = s.get(TaskRow, task_id)
            if row is None:
                raise NotFound(task_id)
            return Task.model_validate_json(row.body)

    def list_for_learner(self, learner_id: str) -> list[Task]:
        with self._sessions() as s:
            rows = s.scalars(select(TaskRow).where(TaskRow.learner_id == learner_id).order_by(TaskRow.created_at))
            return [Task.model_validate_json(r.body) for r in rows]


class SqlEventRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def append(self, event: Event) -> None:
        with self._sessions.begin() as s:
            s.add(EventRow(event_id=event.event_id, task_id=event.task_id, type=event.type, at=event.at,
                           body=event.model_dump_json()))

    def list_for_task(self, task_id: str) -> list[Event]:
        with self._sessions() as s:
            rows = s.scalars(select(EventRow).where(EventRow.task_id == task_id).order_by(EventRow.seq))
            return [Event.model_validate_json(r.body) for r in rows]


class SqlArtifactRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def add(self, artifact: Artifact) -> None:
        with self._sessions.begin() as s:
            s.add(ArtifactRow(
                artifact_id=artifact.artifact_id, task_id=artifact.task_id, name=artifact.name,
                type=artifact.type.value, version=artifact.version, content_hash=artifact.content_hash,
                body=artifact.model_dump_json(),
            ))

    def get(self, artifact_id: str) -> Artifact:
        with self._sessions() as s:
            row = s.get(ArtifactRow, artifact_id)
            if row is None:
                raise NotFound(artifact_id)
            return Artifact.model_validate_json(row.body)

    def list_for_task(self, task_id: str) -> list[Artifact]:
        with self._sessions() as s:
            rows = s.scalars(
                select(ArtifactRow).where(ArtifactRow.task_id == task_id).order_by(ArtifactRow.name, ArtifactRow.version)
            )
            return [Artifact.model_validate_json(r.body) for r in rows]

    def latest(self, task_id: str, name: str) -> Artifact | None:
        with self._sessions() as s:
            row = s.scalars(
                select(ArtifactRow)
                .where(ArtifactRow.task_id == task_id, ArtifactRow.name == name)
                .order_by(ArtifactRow.version.desc())
                .limit(1)
            ).first()
            return Artifact.model_validate_json(row.body) if row else None


class SqlLearnerRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def get(self, learner_id: str) -> LearnerProfile | None:
        with self._sessions() as s:
            row = s.get(LearnerRow, learner_id)
            return LearnerProfile.model_validate_json(row.body) if row else None

    def save(self, profile: LearnerProfile) -> None:
        with self._sessions.begin() as s:
            s.merge(LearnerRow(learner_id=profile.learner_id, updated_at=profile.updated_at,
                               body=profile.model_dump_json()))


class SqlEvidenceRepository:
    """Append-only evidence store: rows are inserted, never updated."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def add(self, evidence: LearningEvidence) -> bool:
        with self._sessions.begin() as s:
            row = s.scalars(select(LearningEvidenceRow).where(
                LearningEvidenceRow.evidence_id == evidence.evidence_id)).first()
            if row is not None:
                if LearningEvidence.model_validate_json(row.body) != evidence:
                    raise EvidenceConflict(
                        f"evidence {evidence.evidence_id} is already recorded with different content")
                return False
            s.add(LearningEvidenceRow(evidence_id=evidence.evidence_id, learner_id=evidence.learner_id,
                                      concept_id=evidence.concept_id, source_type=evidence.source_type,
                                      timestamp=evidence.timestamp, body=evidence.model_dump_json()))
            return True

    def for_learner(self, learner_id: str, concept_id: str | None = None) -> list[LearningEvidence]:
        query = select(LearningEvidenceRow).where(LearningEvidenceRow.learner_id == learner_id)
        if concept_id is not None:
            query = query.where(LearningEvidenceRow.concept_id == concept_id)
        with self._sessions() as s:
            return [LearningEvidence.model_validate_json(r.body)
                    for r in s.scalars(query.order_by(LearningEvidenceRow.seq))]


class SqlLearningEventRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def add(self, event: LearningEvent) -> bool:
        with self._sessions.begin() as s:
            if s.scalars(select(LearningEventRow.seq).where(LearningEventRow.event_id == event.event_id)).first():
                return False
            s.add(LearningEventRow(event_id=event.event_id, learner_id=event.learner_id, type=event.type,
                                   at=event.at, body=event.model_dump_json()))
            return True

    def for_learner(self, learner_id: str) -> list[LearningEvent]:
        with self._sessions() as s:
            rows = s.scalars(select(LearningEventRow).where(LearningEventRow.learner_id == learner_id)
                             .order_by(LearningEventRow.seq))
            return [LearningEvent.model_validate_json(r.body) for r in rows]


class SqlGoalRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def save(self, goal: LearningGoal) -> None:
        with self._sessions.begin() as s:
            s.merge(LearningGoalRow(goal_id=goal.goal_id, learner_id=goal.learner_id, domain=goal.domain,
                                    status=goal.status.value, body=goal.model_dump_json()))

    def get(self, goal_id: str) -> LearningGoal | None:
        with self._sessions() as s:
            row = s.get(LearningGoalRow, goal_id)
            return LearningGoal.model_validate_json(row.body) if row else None

    def for_learner(self, learner_id: str) -> list[LearningGoal]:
        with self._sessions() as s:
            rows = s.scalars(select(LearningGoalRow).where(LearningGoalRow.learner_id == learner_id)
                             .order_by(LearningGoalRow.goal_id))
            return [LearningGoal.model_validate_json(r.body) for r in rows]


class SqlCurriculumRepository:
    """Curricula: current pointers, append-only immutable versions and the last progress snapshot."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def save_record(self, record: CurriculumRecord) -> None:
        with self._sessions.begin() as s:
            s.merge(CurriculumRow(curriculum_id=record.curriculum_id, learner_id=record.learner_id,
                                  goal_id=record.goal_id, body=record.model_dump_json()))

    def record(self, curriculum_id: str) -> CurriculumRecord | None:
        with self._sessions() as s:
            row = s.get(CurriculumRow, curriculum_id)
            return CurriculumRecord.model_validate_json(row.body) if row else None

    def records_for_learner(self, learner_id: str) -> list[CurriculumRecord]:
        with self._sessions() as s:
            rows = s.scalars(select(CurriculumRow).where(CurriculumRow.learner_id == learner_id)
                             .order_by(CurriculumRow.curriculum_id))
            return [CurriculumRecord.model_validate_json(r.body) for r in rows]

    def add_version(self, version: CurriculumVersion) -> bool:
        with self._sessions.begin() as s:
            row = s.get(CurriculumVersionRow, version.version_id)
            if row is not None:
                existing = CurriculumVersion.model_validate_json(row.body)
                if existing.model_dump(exclude={"created_at"}) != version.model_dump(exclude={"created_at"}):
                    raise VersionConflict(f"curriculum version {version.version_id} exists with different content")
                return False
            s.add(CurriculumVersionRow(version_id=version.version_id, curriculum_id=version.curriculum_id,
                                       version=version.version, content_hash=version.content_hash,
                                       body=version.model_dump_json()))
            return True

    def version(self, version_id: str) -> CurriculumVersion | None:
        with self._sessions() as s:
            row = s.get(CurriculumVersionRow, version_id)
            return CurriculumVersion.model_validate_json(row.body) if row else None

    def versions(self, curriculum_id: str) -> list[CurriculumVersion]:
        with self._sessions() as s:
            rows = s.scalars(select(CurriculumVersionRow).where(CurriculumVersionRow.curriculum_id == curriculum_id)
                             .order_by(CurriculumVersionRow.version))
            return [CurriculumVersion.model_validate_json(r.body) for r in rows]

    def save_progress(self, progress: CurriculumProgress) -> None:
        with self._sessions.begin() as s:
            s.merge(CurriculumProgressRow(curriculum_id=progress.curriculum_id, body=progress.model_dump_json()))

    def progress(self, curriculum_id: str) -> CurriculumProgress | None:
        with self._sessions() as s:
            row = s.get(CurriculumProgressRow, curriculum_id)
            return CurriculumProgress.model_validate_json(row.body) if row else None
