"""SQL repositories: persistence only, no business rules."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.schemas.artifact import Artifact
from app.schemas.events import Event
from app.schemas.learner import LearnerProfile
from app.schemas.task import Task
from app.storage.orm import ArtifactRow, EventRow, LearnerRow, TaskRow


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
