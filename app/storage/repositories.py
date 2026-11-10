"""SQL repositories: persistence only, no business rules."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.observability.spend import abandoned

from app.schemas.artifact import Artifact
from app.schemas.assessment import (
    AssessmentAttempt,
    AssessmentGrade,
    AssessmentItem,
    AssessmentRubric,
    AttemptExists,
    AttemptOutcome,
    ItemConflict,
)
from app.schemas.classroom import ClassMember, Classroom
from app.schemas.curriculum import (
    CurriculumProgress,
    CurriculumRecord,
    CurriculumVersion,
    RecordConflict,
    VersionConflict,
)
from app.schemas.common import utcnow
from app.schemas.events import Event
from app.schemas.lease import Lease, LeaseLost, WriteConflict, write_fence
from app.schemas.learning_cycle import (
    TERMINAL_CYCLE_STATUSES,
    CycleConflict,
    CycleLeaseLost,
    CycleRequestRecord,
    LearningCycle,
)
from app.schemas.learner import (
    EvidenceConflict,
    LearnerProfile,
    LearningEvent,
    LearningEvidence,
    LearningGoal,
    ProfileConflict,
)
from app.schemas.spend import (
    CounterTotals,
    ReservationRefused,
    SettleResult,
    SpendEntry,
    SpendInFlight,
    SpendReservation,
    SpendSettlement,
    SpendState,
)
from app.schemas.job import ACTIVE_JOB_STATUSES, JobErrorCategory, JobKind, JobStatus, TaskJob
from app.schemas.task import Task, TaskControl, TaskStatus
from app.schemas.teaching import (
    InteractionEvidence,
    OutboxItem,
    SessionChange,
    SessionConflict,
    TeachingRequestRecord,
    TeachingSession,
    TeachingTurn,
)
from app.storage.orm import (
    ArtifactRow,
    AssessmentAttemptRow,
    AssessmentGradeRow,
    AssessmentItemRow,
    AssessmentRubricRow,
    ClassMemberRow,
    ClassRow,
    CurriculumProgressRow,
    CurriculumRow,
    CurriculumVersionRow,
    EventRow,
    LearnerRow,
    LearningEventRow,
    LearningEvidenceRow,
    LearningCycleRequestRow,
    LearningCycleRow,
    LearningCycleSlotRow,
    LearningGoalRow,
    SpendCounterRow,
    SpendEntryRow,
    TaskJobRow,
    TaskRow,
    InteractionEvidenceRow,
    TeachingOutboxRow,
    TeachingRequestRow,
    TeachingSessionRow,
    TeachingTurnRow,
)

logger = logging.getLogger(__name__)


class NotFound(KeyError):
    pass


Clock = Callable[[], datetime]


def _ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def _require_fence(s: Session, lease: Lease, now: datetime) -> None:
    """Inside the caller's transaction: refuse the write unless `lease` is still the task's current, unexpired lease.
    The check is itself a (no-op) UPDATE of the task row, so it takes the row's write lock and nobody can claim the
    task between this check and the caller's commit."""
    if lease.resource != "task":
        raise ValueError(f"no fence check for resource {lease.resource}")
    checked = s.execute(update(TaskRow).where(
        TaskRow.task_id == lease.resource_id, TaskRow.lease_fence == lease.fence, TaskRow.lease_owner == lease.owner,
        TaskRow.lease_expires_ms > _ms(now)).values(lease_fence=TaskRow.lease_fence))
    if checked.rowcount != 1:
        raise LeaseLost(lease)


def _fence_current(s: Session, clock: Clock) -> None:
    """Inside a learner-memory or curriculum write's transaction: when the write is made under a task's execution
    (`fenced_writes`), refuse it unless that lease is still current and unexpired (see `_require_fence`)."""
    lease = write_fence()
    if lease is not None:
        _require_fence(s, lease, clock())


def _versioned_save(s: Session, row_type, key_column, key: str, stored: int | None, values: dict, conflict) -> int:
    """Insert (`stored` None) or update the row only at version `stored`, raising its version by one. Raises
    `conflict` (nothing written) when the row exists already, or moved past `stored`. Returns the new version."""
    if stored is None:
        if s.get(row_type, key) is not None:
            raise conflict
        s.add(row_type(**{key_column.key: key}, version=1, **values))
        try:
            s.flush()
        except IntegrityError as exc:  # a concurrent insert of the same key won
            raise conflict from exc
        return 1
    done = s.execute(update(row_type).where(key_column == key, row_type.version == stored)
                     .values(version=stored + 1, **values))
    if done.rowcount != 1:
        raise conflict
    return stored + 1


class SqlTaskRepository:
    """Tasks, their execution lease and their pause/cancel requests.

    Two kinds of write: a leased write (`save(task, lease=...)`) by the worker executing the task, applied only while
    its lease is current and unexpired; and an unleased write (`save(task)`), version-checked against `task.version`
    and refused while any worker holds a live lease. Both raise the version, so neither silently overwrites the other.
    """

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def create(self, task: Task) -> bool:
        """Insert a new task; False (nothing written) when the id exists."""
        try:
            with self._sessions.begin() as s:
                if s.get(TaskRow, task.task_id) is not None:
                    return False
                task.version = 1
                s.add(self._row(task))
                s.flush()
        except IntegrityError:
            return False
        return True

    def save(self, task: Task, *, lease: Lease | None = None, clear_control: Iterable[str] = ()) -> None:
        """Write the task's body and status and raise its version. `clear_control` names requests ("pause",
        "cancel") the holder has acted on. Raises LeaseLost (leased) or WriteConflict (unleased); nothing is written
        then."""
        now = self._clock()
        body = task.model_copy(update={"version": task.version + 1})
        values: dict = {"status": task.status.value, "updated_at": task.updated_at, "body": body.model_dump_json(),
                        "version": TaskRow.version + 1}
        for name in clear_control:
            values[{"pause": "pause_requested", "cancel": "cancel_requested"}[name]] = False
        with self._sessions.begin() as s:
            if lease is not None:
                if lease.resource_id != task.task_id:
                    raise ValueError("the lease is for another task")
                done = s.execute(update(TaskRow).where(
                    TaskRow.task_id == task.task_id, TaskRow.lease_fence == lease.fence,
                    TaskRow.lease_owner == lease.owner, TaskRow.lease_expires_ms > _ms(now)).values(**values))
                if done.rowcount != 1:
                    raise LeaseLost(lease)
            elif s.get(TaskRow, task.task_id) is None:
                task.version = 1
                s.add(self._row(task))
                return
            else:
                done = s.execute(update(TaskRow).where(
                    TaskRow.task_id == task.task_id, TaskRow.version == task.version,
                    or_(TaskRow.lease_owner.is_(None), TaskRow.lease_expires_ms <= _ms(now))).values(**values))
                if done.rowcount != 1:
                    raise WriteConflict(f"task {task.task_id} changed concurrently or is being executed "
                                        f"(expected version {task.version})")
        task.version += 1

    def get(self, task_id: str) -> Task:
        with self._sessions() as s:
            row = s.get(TaskRow, task_id)
            if row is None:
                raise NotFound(task_id)
            return self._task(row)

    def list_for_learner(self, learner_id: str) -> list[Task]:
        with self._sessions() as s:
            rows = s.scalars(select(TaskRow).where(TaskRow.learner_id == learner_id).order_by(TaskRow.created_at))
            return [self._task(r) for r in rows]

    # --- the execution lease --------------------------------------------------------------------------------------

    def claim(self, task_id: str, owner: str, ttl: timedelta) -> Lease | None:
        """Take the task's lease when nobody holds a live one, raising its fencing generation: every earlier holder
        is stale from this commit on. None while another worker's lease runs."""
        now = self._clock()
        until = now + ttl
        with self._sessions.begin() as s:
            done = s.execute(update(TaskRow).where(
                TaskRow.task_id == task_id,
                or_(TaskRow.lease_owner.is_(None), TaskRow.lease_expires_ms <= _ms(now)))
                .values(lease_owner=owner, lease_fence=TaskRow.lease_fence + 1, lease_expires_ms=_ms(until)))
            if done.rowcount != 1:
                if s.get(TaskRow, task_id) is None:
                    raise NotFound(task_id)
                return None
            fence = s.scalar(select(TaskRow.lease_fence).where(TaskRow.task_id == task_id))
        return Lease(resource_id=task_id, owner=owner, fence=fence, until=until)

    def renew(self, lease: Lease, ttl: timedelta) -> Lease | None:
        """Extend a lease that is still current and unexpired; None when it was lost (an expired lease is never
        renewed: it must be claimed again, which makes its old holder stale)."""
        now = self._clock()
        until = now + ttl
        with self._sessions.begin() as s:
            done = s.execute(update(TaskRow).where(
                TaskRow.task_id == lease.resource_id, TaskRow.lease_fence == lease.fence,
                TaskRow.lease_owner == lease.owner, TaskRow.lease_expires_ms > _ms(now))
                .values(lease_expires_ms=_ms(until)))
        return lease.model_copy(update={"until": until}) if done.rowcount == 1 else None

    def release(self, lease: Lease) -> bool:
        """Give the lease back (the generation stays). False when it had already been taken over."""
        with self._sessions.begin() as s:
            done = s.execute(update(TaskRow).where(
                TaskRow.task_id == lease.resource_id, TaskRow.lease_fence == lease.fence,
                TaskRow.lease_owner == lease.owner).values(lease_owner=None, lease_expires_ms=None))
        return done.rowcount == 1

    def holds(self, lease: Lease) -> bool:
        """Whether `lease` is still the task's current, unexpired lease (a pre-check; writes check atomically)."""
        with self._sessions() as s:
            row = s.get(TaskRow, lease.resource_id)
            return (row is not None and row.lease_fence == lease.fence and row.lease_owner == lease.owner
                    and row.lease_expires_ms is not None and row.lease_expires_ms > _ms(self._clock()))

    def lease(self, task_id: str) -> Lease | None:
        """The live lease on the task, if any (diagnostics)."""
        with self._sessions() as s:
            row = s.get(TaskRow, task_id)
            if row is None:
                raise NotFound(task_id)
            if row.lease_owner is None or row.lease_expires_ms is None or row.lease_expires_ms <= _ms(self._clock()):
                return None
            until = datetime.fromtimestamp(row.lease_expires_ms / 1000, tz=self._clock().tzinfo)
            return Lease(resource_id=task_id, owner=row.lease_owner, fence=row.lease_fence, until=until)

    def stranded(self, statuses: Iterable[TaskStatus], *, idle_before: datetime, limit: int = 100) -> list[str]:
        """Tasks in one of `statuses` that nobody is executing and nothing will continue: their worker's lease ran
        out (it died or hung), or no lease is held and the task has not changed since `idle_before`, and they have no
        active (queued or running) job. Oldest first."""
        now_ms = _ms(self._clock())
        active_job = select(TaskJobRow.job_id).where(TaskJobRow.task_id == TaskRow.task_id,
                                                     TaskJobRow.status.in_(_ACTIVE)).exists()
        with self._sessions() as s:
            return list(s.scalars(select(TaskRow.task_id).where(
                TaskRow.status.in_([st.value for st in statuses]),
                or_(TaskRow.lease_expires_ms <= now_ms,
                    and_(TaskRow.lease_owner.is_(None), TaskRow.updated_at <= idle_before)),
                ~active_job).order_by(TaskRow.updated_at, TaskRow.task_id).limit(limit)))

    # --- pause and cancel requests from other callers -------------------------------------------------------------

    def control(self, task_id: str) -> TaskControl:
        with self._sessions() as s:
            row = s.get(TaskRow, task_id)
            if row is None:
                raise NotFound(task_id)
            return TaskControl(pause_requested=bool(row.pause_requested), cancel_requested=bool(row.cancel_requested))

    def request_control(self, task_id: str, *, pause: bool = False, cancel: bool = False) -> None:
        """Set a request flag in one statement: never lost to, nor overwritten by, the executing worker's writes."""
        values = {k: True for k, on in (("pause_requested", pause), ("cancel_requested", cancel)) if on}
        if values:
            with self._sessions.begin() as s:
                s.execute(update(TaskRow).where(TaskRow.task_id == task_id).values(**values))

    @staticmethod
    def _row(task: Task) -> TaskRow:
        return TaskRow(task_id=task.task_id, user_id=task.user_id, learner_id=task.learner_id,
                       status=task.status.value, created_at=task.created_at, updated_at=task.updated_at,
                       body=task.model_dump_json(), version=task.version, lease_fence=0,
                       pause_requested=task.control.pause_requested, cancel_requested=task.control.cancel_requested)

    @staticmethod
    def _task(row: TaskRow) -> Task:
        task = Task.model_validate_json(row.body)
        task.version = row.version
        task.control = TaskControl(pause_requested=bool(row.pause_requested),
                                   cancel_requested=bool(row.cancel_requested))
        return task


class SqlEventRepository:
    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def append(self, event: Event, fence: Lease | None = None) -> None:
        """Store an event once. With `fence` (written by a task's worker) only while that worker's lease is current;
        otherwise LeaseLost and nothing is stored."""
        with self._sessions.begin() as s:
            if fence is not None:
                _require_fence(s, fence, self._clock())
            if s.scalar(select(EventRow.seq).where(EventRow.event_id == event.event_id)) is not None:
                return  # an event with a stable id published again (e.g. after a restart) is stored once
            s.add(EventRow(event_id=event.event_id, task_id=event.task_id, type=event.type, at=event.at,
                           body=event.model_dump_json()))

    def list_for_task(self, task_id: str) -> list[Event]:
        with self._sessions() as s:
            rows = s.scalars(select(EventRow).where(EventRow.task_id == task_id).order_by(EventRow.seq))
            return [Event.model_validate_json(r.body) for r in rows]


class SqlArtifactRepository:
    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def add(self, artifact: Artifact, fence: Lease | None = None) -> None:
        """Record an artifact version. With `fence` (stored by a task's worker) only while that worker's lease is
        current: LeaseLost otherwise. A version another writer already recorded is a WriteConflict."""
        try:
            with self._sessions.begin() as s:
                if fence is not None:
                    _require_fence(s, fence, self._clock())
                self._insert(s, artifact)
        except IntegrityError as exc:
            raise WriteConflict(f"artifact {artifact.task_id}/{artifact.name} v{artifact.version} already exists") \
                from exc

    @staticmethod
    def _insert(s: Session, artifact: Artifact) -> None:
        s.add(ArtifactRow(
            artifact_id=artifact.artifact_id, task_id=artifact.task_id, name=artifact.name,
            type=artifact.type.value, version=artifact.version, content_hash=artifact.content_hash,
            body=artifact.model_dump_json(),
        ))
        s.flush()

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
    """Learner profiles. A profile is one JSON document rewritten by every change, so each save is version-checked
    against the read it was made from (`ProfileConflict` instead of a lost update) and, under a task's execution,
    fenced by the task's lease."""

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def get(self, learner_id: str) -> LearnerProfile | None:
        with self._sessions() as s:
            row = s.get(LearnerRow, learner_id)
            if row is None:
                return None
            profile = LearnerProfile.model_validate_json(row.body)
            profile._row_version = row.version
            return profile

    def save(self, profile: LearnerProfile) -> None:
        conflict = ProfileConflict(f"learner {profile.learner_id} changed since it was read")
        with self._sessions.begin() as s:
            _fence_current(s, self._clock)
            version = _versioned_save(s, LearnerRow, LearnerRow.learner_id, profile.learner_id, profile._row_version,
                                      {"updated_at": profile.updated_at, "body": profile.model_dump_json()}, conflict)
        profile._row_version = version


class SqlEvidenceRepository:
    """Append-only evidence store: rows are inserted, never updated."""

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def add(self, evidence: LearningEvidence) -> bool:
        with self._sessions.begin() as s:
            _fence_current(s, self._clock)
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
    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def add(self, event: LearningEvent) -> bool:
        with self._sessions.begin() as s:
            _fence_current(s, self._clock)
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
    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def save(self, goal: LearningGoal) -> None:
        with self._sessions.begin() as s:
            _fence_current(s, self._clock)
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
    """Curricula: current pointers, append-only immutable versions and the last progress snapshot. A pointer save is
    version-checked against the read it was made from (`RecordConflict`); under a task's execution every write is
    fenced by the task's lease."""

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def save_record(self, record: CurriculumRecord) -> None:
        conflict = RecordConflict(f"curriculum {record.curriculum_id} changed since it was read")
        with self._sessions.begin() as s:
            _fence_current(s, self._clock)
            version = _versioned_save(s, CurriculumRow, CurriculumRow.curriculum_id, record.curriculum_id,
                                      record._row_version, {"learner_id": record.learner_id,
                                                            "goal_id": record.goal_id,
                                                            "body": record.model_dump_json()}, conflict)
        record._row_version = version

    def record(self, curriculum_id: str) -> CurriculumRecord | None:
        with self._sessions() as s:
            row = s.get(CurriculumRow, curriculum_id)
            return self._record(row) if row else None

    def records_for_learner(self, learner_id: str) -> list[CurriculumRecord]:
        with self._sessions() as s:
            rows = s.scalars(select(CurriculumRow).where(CurriculumRow.learner_id == learner_id)
                             .order_by(CurriculumRow.curriculum_id))
            return [self._record(r) for r in rows]

    @staticmethod
    def _record(row: CurriculumRow) -> CurriculumRecord:
        record = CurriculumRecord.model_validate_json(row.body)
        record._row_version = row.version
        return record

    def add_version(self, version: CurriculumVersion) -> bool:
        with self._sessions.begin() as s:
            _fence_current(s, self._clock)
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
            _fence_current(s, self._clock)
            s.merge(CurriculumProgressRow(curriculum_id=progress.curriculum_id, body=progress.model_dump_json()))

    def progress(self, curriculum_id: str) -> CurriculumProgress | None:
        with self._sessions() as s:
            row = s.get(CurriculumProgressRow, curriculum_id)
            return CurriculumProgress.model_validate_json(row.body) if row else None


class SqlTeachingRepository:
    """Teaching sessions, turns, interaction evidence, applied requests and the outbox. Every change is one
    transaction, applied only against the session version it was computed from."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create(self, change: SessionChange) -> bool:
        try:
            with self._sessions.begin() as s:
                if s.get(TeachingSessionRow, change.session.session_id) is not None:
                    return False
                s.add(self._session_row(change.session))
                self._add(s, change)
        except IntegrityError:
            return False
        return True

    def apply(self, change: SessionChange, expected_version: int, request: TeachingRequestRecord | None = None) -> None:
        session = change.session
        try:
            with self._sessions.begin() as s:
                updated = s.execute(
                    update(TeachingSessionRow)
                    .where(TeachingSessionRow.session_id == session.session_id,
                           TeachingSessionRow.version == expected_version)
                    .values(version=session.version, status=session.status.value, body=session.model_dump_json()))
                if updated.rowcount != 1:
                    raise SessionConflict(f"session {session.session_id} changed concurrently "
                                          f"(expected version {expected_version})")
                self._add(s, change)
                if request is not None:
                    s.add(TeachingRequestRow(session_id=request.session_id, client_turn_id=request.client_turn_id,
                                             body=request.model_dump_json()))
                s.flush()
        except IntegrityError as exc:
            raise SessionConflict(f"session {session.session_id}: the turn or request was already stored") from exc

    @staticmethod
    def _session_row(session: TeachingSession) -> TeachingSessionRow:
        return TeachingSessionRow(session_id=session.session_id, learner_id=session.learner_id,
                                  lesson_id=session.lesson_id, status=session.status.value, version=session.version,
                                  body=session.model_dump_json())

    @staticmethod
    def _add(s: Session, change: SessionChange) -> None:
        sid = change.session.session_id
        for turn in change.turns:
            s.add(TeachingTurnRow(turn_id=turn.turn_id, session_id=sid, sequence=turn.sequence,
                                  body=turn.model_dump_json()))
        for ev in change.evidence:
            s.add(InteractionEvidenceRow(evidence_id=ev.evidence_id, session_id=sid, body=ev.model_dump_json()))
        for item in change.outbox:
            s.add(TeachingOutboxRow(item_id=item.item_id, session_id=sid, published=False,
                                    body=item.model_dump_json()))

    def get(self, session_id: str) -> TeachingSession | None:
        with self._sessions() as s:
            row = s.get(TeachingSessionRow, session_id)
            return TeachingSession.model_validate_json(row.body) if row else None

    def for_learner(self, learner_id: str) -> list[TeachingSession]:
        with self._sessions() as s:
            rows = s.scalars(select(TeachingSessionRow).where(TeachingSessionRow.learner_id == learner_id))
            return sorted((TeachingSession.model_validate_json(r.body) for r in rows),
                          key=lambda x: (x.started_at, x.session_id))

    def turns(self, session_id: str) -> list[TeachingTurn]:
        with self._sessions() as s:
            rows = s.scalars(select(TeachingTurnRow).where(TeachingTurnRow.session_id == session_id)
                             .order_by(TeachingTurnRow.sequence))
            return [TeachingTurn.model_validate_json(r.body) for r in rows]

    def evidence(self, session_id: str) -> list[InteractionEvidence]:
        with self._sessions() as s:
            rows = s.scalars(select(InteractionEvidenceRow).where(InteractionEvidenceRow.session_id == session_id)
                             .order_by(InteractionEvidenceRow.seq))
            return [InteractionEvidence.model_validate_json(r.body) for r in rows]

    def request(self, session_id: str, client_turn_id: str) -> TeachingRequestRecord | None:
        with self._sessions() as s:
            row = s.get(TeachingRequestRow, (session_id, client_turn_id))
            return TeachingRequestRecord.model_validate_json(row.body) if row else None

    def pending_outbox(self, session_id: str) -> list[OutboxItem]:
        with self._sessions() as s:
            rows = s.scalars(select(TeachingOutboxRow).where(TeachingOutboxRow.session_id == session_id,
                                                             TeachingOutboxRow.published.is_(False))
                             .order_by(TeachingOutboxRow.seq))
            return [OutboxItem.model_validate_json(r.body) for r in rows]

    def mark_published(self, item_id: str) -> None:
        with self._sessions.begin() as s:
            s.execute(update(TeachingOutboxRow).where(TeachingOutboxRow.item_id == item_id).values(published=True))


class SqlAssessmentRepository:
    """Assessment items and rubrics (immutable), attempts and their grades (append-only, stored together)."""

    ATTEMPT_RETRIES = 5  # a concurrent attempt took the same attempt number: take the next one

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def _save(self, row_type, key: str, value, **columns) -> bool:
        digest = value.content_hash()
        try:
            with self._sessions.begin() as s:
                current = s.get(row_type, key)
                if current is None:
                    s.add(row_type(**columns, content_hash=digest, body=value.model_dump_json()))
                    return True
        except IntegrityError:
            pass
        with self._sessions() as s:
            current = s.get(row_type, key)
        if current is None or current.content_hash != digest:
            raise ItemConflict(f"{key} is already stored with different content")
        return False

    def save_item(self, item: AssessmentItem) -> bool:
        return self._save(AssessmentItemRow, item.assessment_item_id, item, item_id=item.assessment_item_id,
                          lesson_id=item.lesson_id)

    def item(self, item_id: str) -> AssessmentItem | None:
        with self._sessions() as s:
            row = s.get(AssessmentItemRow, item_id)
            return AssessmentItem.model_validate_json(row.body) if row else None

    def items_for_lesson(self, lesson_ids: Iterable[str]) -> list[AssessmentItem]:
        ids = sorted(set(lesson_ids))
        with self._sessions() as s:
            rows = s.scalars(select(AssessmentItemRow).where(AssessmentItemRow.lesson_id.in_(ids))
                             .order_by(AssessmentItemRow.item_id))
            return [AssessmentItem.model_validate_json(r.body) for r in rows]

    def save_rubric(self, rubric: AssessmentRubric) -> bool:
        return self._save(AssessmentRubricRow, rubric.rubric_id, rubric, rubric_id=rubric.rubric_id)

    def rubric(self, rubric_id: str) -> AssessmentRubric | None:
        with self._sessions() as s:
            row = s.get(AssessmentRubricRow, rubric_id)
            return AssessmentRubric.model_validate_json(row.body) if row else None

    def add_attempt(self, attempt: AssessmentAttempt, grade: AssessmentGrade,
                    fence: Lease | None = None) -> AssessmentAttempt:
        """Both rows in one transaction. With `fence` (a task worker grading), only while that worker's lease is
        current: the check is the transaction's first statement, so a refused write (LeaseLost) stores neither row."""
        for _ in range(self.ATTEMPT_RETRIES):
            try:
                with self._sessions.begin() as s:
                    if fence is not None:
                        _require_fence(s, fence, self._clock())
                    if s.get(AssessmentAttemptRow, attempt.attempt_id) is not None:
                        raise AttemptExists(attempt.attempt_id)
                    count = s.scalar(select(func.count()).select_from(AssessmentAttemptRow).where(
                        AssessmentAttemptRow.item_id == attempt.assessment_item_id,
                        AssessmentAttemptRow.learner_id == attempt.learner_id)) or 0
                    stored = attempt.model_copy(update={"attempt_number": count + 1})
                    s.add(AssessmentAttemptRow(attempt_id=stored.attempt_id, item_id=stored.assessment_item_id,
                                               learner_id=stored.learner_id, attempt_number=stored.attempt_number,
                                               grade_id=grade.grade_id, body=stored.model_dump_json()))
                    s.add(AssessmentGradeRow(grade_id=grade.grade_id, attempt_id=grade.attempt_id,
                                             body=grade.model_dump_json()))
                    s.flush()
                return stored
            except IntegrityError:
                if self.attempt(attempt.attempt_id) is not None:
                    raise AttemptExists(attempt.attempt_id) from None
        raise AttemptExists(f"{attempt.attempt_id}: no attempt number available after concurrent attempts")

    def attempt(self, attempt_id: str) -> AssessmentAttempt | None:
        with self._sessions() as s:
            row = s.get(AssessmentAttemptRow, attempt_id)
            return AssessmentAttempt.model_validate_json(row.body) if row else None

    def attempts(self, item_id: str, learner_id: str) -> list[AssessmentAttempt]:
        with self._sessions() as s:
            rows = s.scalars(select(AssessmentAttemptRow).where(AssessmentAttemptRow.item_id == item_id,
                                                                AssessmentAttemptRow.learner_id == learner_id)
                             .order_by(AssessmentAttemptRow.attempt_number))
            return [AssessmentAttempt.model_validate_json(r.body) for r in rows]

    def grade(self, grade_id: str) -> AssessmentGrade | None:
        with self._sessions() as s:
            row = s.get(AssessmentGradeRow, grade_id)
            return AssessmentGrade.model_validate_json(row.body) if row else None

    def complete(self, attempt_id: str, outcome: AttemptOutcome | None, at,
                 fence: Lease | None = None) -> AssessmentAttempt:
        with self._sessions.begin() as s:
            if fence is not None:
                _require_fence(s, fence, self._clock())
            row = s.get(AssessmentAttemptRow, attempt_id)
            if row is None:
                raise KeyError(attempt_id)
            current = AssessmentAttempt.model_validate_json(row.body)
            if current.completed_at is None:
                current = current.model_copy(update={"completed_at": at, "outcome": outcome})
                row.body = current.model_dump_json()
            return current


def _cycle_lease_columns(cycle: LearningCycle) -> dict:
    lease = cycle.lease
    return {"lease_token": lease.token if lease else None, "lease_expires_ms": _ms(lease.until) if lease else None}


class SqlLearningCycleRepository:
    """Learning cycles, the one-active-cycle-per-learner slot and received responses. Every change is one transaction,
    applied only against the cycle version it was computed from."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create(self, cycle: LearningCycle) -> bool:
        try:
            with self._sessions.begin() as s:
                if s.get(LearningCycleRow, cycle.cycle_id) is not None:
                    return False
                s.add(LearningCycleRow(cycle_id=cycle.cycle_id, learner_id=cycle.learner_id, status=cycle.status.value,
                                       version=cycle.version, created_at=cycle.created_at,
                                       body=cycle.model_dump_json(), **_cycle_lease_columns(cycle)))
                s.add(LearningCycleSlotRow(learner_id=cycle.learner_id, cycle_id=cycle.cycle_id))
                s.flush()
        except IntegrityError:
            if self.get(cycle.cycle_id) is not None:
                return False  # the same key, created concurrently
            holder = self.active_for(cycle.learner_id)
            raise CycleConflict(f"learner has an active learning cycle {holder.cycle_id if holder else '?'}") \
                from None
        return True

    def apply(self, cycle: LearningCycle, expected_version: int, request: CycleRequestRecord | None = None, *,
              lease_token: str | None = None, at: datetime | None = None) -> None:
        """Write the change if the stored cycle is still `expected_version`. With `lease_token` (a driver's write) the
        same statement also requires that lease to be the stored one and unexpired at `at`: CycleLeaseLost otherwise,
        so a driver whose lease ran out or was taken over commits nothing."""
        where = [LearningCycleRow.cycle_id == cycle.cycle_id, LearningCycleRow.version == expected_version]
        if lease_token is not None:
            where += [LearningCycleRow.lease_token == lease_token,
                      LearningCycleRow.lease_expires_ms > _ms(at or utcnow())]
        try:
            with self._sessions.begin() as s:
                updated = s.execute(
                    update(LearningCycleRow).where(*where)
                    .values(version=cycle.version, status=cycle.status.value, body=cycle.model_dump_json(),
                            **_cycle_lease_columns(cycle)))
                if updated.rowcount != 1:
                    if lease_token is not None and s.scalar(select(LearningCycleRow.version).where(
                            LearningCycleRow.cycle_id == cycle.cycle_id)) == expected_version:
                        raise CycleLeaseLost(f"cycle {cycle.cycle_id}: the drive lease expired or was taken over")
                    raise CycleConflict(f"cycle {cycle.cycle_id} changed concurrently "
                                        f"(expected version {expected_version})")
                if cycle.status in TERMINAL_CYCLE_STATUSES:
                    s.execute(delete(LearningCycleSlotRow).where(LearningCycleSlotRow.cycle_id == cycle.cycle_id))
                if request is not None:
                    s.add(LearningCycleRequestRow(cycle_id=request.cycle_id,
                                                  client_response_id=request.client_response_id,
                                                  body=request.model_dump_json()))
                s.flush()
        except IntegrityError as exc:
            raise CycleConflict(f"cycle {cycle.cycle_id}: the response was already received") from exc

    def get(self, cycle_id: str) -> LearningCycle | None:
        with self._sessions() as s:
            row = s.get(LearningCycleRow, cycle_id)
            return LearningCycle.model_validate_json(row.body) if row else None

    def active_for(self, learner_id: str) -> LearningCycle | None:
        with self._sessions() as s:
            slot = s.get(LearningCycleSlotRow, learner_id)
            cycle_id = slot.cycle_id if slot else None
        return self.get(cycle_id) if cycle_id else None

    def for_learner(self, learner_id: str) -> list[LearningCycle]:
        with self._sessions() as s:
            rows = s.scalars(select(LearningCycleRow).where(LearningCycleRow.learner_id == learner_id)
                             .order_by(LearningCycleRow.created_at.desc(), LearningCycleRow.cycle_id.desc()))
            return [LearningCycle.model_validate_json(r.body) for r in rows]

    def request(self, cycle_id: str, client_response_id: str) -> CycleRequestRecord | None:
        with self._sessions() as s:
            row = s.get(LearningCycleRequestRow, (cycle_id, client_response_id))
            return CycleRequestRecord.model_validate_json(row.body) if row else None

    def mark_applied(self, cycle_id: str, client_response_id: str) -> None:
        with self._sessions.begin() as s:
            row = s.get(LearningCycleRequestRow, (cycle_id, client_response_id))
            if row is not None:
                record = CycleRequestRecord.model_validate_json(row.body)
                record.applied = True
                row.body = record.model_dump_json()

    def forget_request(self, cycle_id: str, client_response_id: str) -> None:
        with self._sessions.begin() as s:
            s.execute(delete(LearningCycleRequestRow).where(LearningCycleRequestRow.cycle_id == cycle_id,
                                                            LearningCycleRequestRow.client_response_id ==
                                                            client_response_id))


# --- the spend ledger ----------------------------------------------------------------------------------------------

_RETRYABLE_SQLSTATES = {"40001", "40P01"}  # PostgreSQL serialization failure, deadlock


def _retryable(exc: DBAPIError) -> bool:
    code = getattr(exc.orig, "pgcode", None) or getattr(exc.orig, "sqlstate", None)
    return code in _RETRYABLE_SQLSTATES or "database is locked" in str(exc.orig)


def _ensure_counter(s: Session, scope_key: str, metric: str, now: datetime, settled: int = 0) -> None:
    """Create the counter row if it does not exist (never overwrites one: concurrent creators both succeed)."""
    values = {"scope_key": scope_key, "metric": metric, "reserved": 0, "settled": settled, "updated_at": now}
    dialect = s.get_bind().dialect.name
    if dialect in ("sqlite", "postgresql"):
        insert = sqlite_insert if dialect == "sqlite" else postgresql_insert
        s.execute(insert(SpendCounterRow).values(**values).on_conflict_do_nothing(
            index_elements=["scope_key", "metric"]))
        return
    try:  # other databases: a savepoint around a plain insert
        with s.begin_nested():
            s.add(SpendCounterRow(**values))
    except IntegrityError:
        pass


class SqlSpendRepository:
    """The spend ledger and its budget counters (`SpendStore`).

    Reservation is one transaction: the task lease check (when fenced), then one conditional increment per counter
    (`settled + reserved + check <= limit` in the UPDATE's own WHERE, so concurrent reservations can never both pass
    on the same old value), in sorted (scope, metric) order, then the entry insert. Settlement is one transaction too:
    a conditional state change of the entry (`WHERE state = 'reserved'`, or 'unknown' for a late real result) and the
    counter adjustments; a repeated settlement matches no row and changes nothing. Settlement is never fenced.

    Every transaction starts with a write (SQLite: no read-to-write lock upgrade). PostgreSQL deadlocks and
    serialization failures, and SQLite's "database is locked", are retried a bounded number of times.
    """

    TRIES = 3

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def reserve(self, reservation: SpendReservation, *, fence: Lease | None, now: datetime) -> None:
        r = reservation
        for attempt in range(1, self.TRIES + 1):
            try:
                with self._sessions.begin() as s:
                    if fence is not None:
                        _require_fence(s, fence, now)  # a worker that lost its task lease cannot start a paid call
                    for c in sorted(r.charges, key=lambda c: (c.scope_key, c.metric)):
                        _ensure_counter(s, c.scope_key, c.metric, now)
                        where = [SpendCounterRow.scope_key == c.scope_key, SpendCounterRow.metric == c.metric]
                        if c.limit is not None:
                            where.append(SpendCounterRow.settled + SpendCounterRow.reserved + c.check <= c.limit)
                        done = s.execute(update(SpendCounterRow).where(*where).values(
                            reserved=SpendCounterRow.reserved + c.amount, updated_at=now))
                        if done.rowcount != 1:
                            row = s.get(SpendCounterRow, (c.scope_key, c.metric))
                            raise ReservationRefused(c, row.reserved if row else 0, row.settled if row else 0)
                    s.add(SpendEntryRow(
                        entry_id=r.entry_id, operation_id=r.operation_id, attempt=r.attempt,
                        idempotency_key=r.idempotency_key, capability=r.capability, operation=r.operation,
                        provider=r.provider, model=r.model, task_id=r.task_id, session_id=r.session_id,
                        attempt_ref=r.attempt_ref, learner_id=r.learner_id, node_id=r.node_id, agent_id=r.agent_id,
                        worker=r.worker, lease_fence=r.lease_fence,
                        scope_keys=json.dumps(sorted({c.scope_key for c in r.charges})),
                        state=SpendState.RESERVED.value, estimate=json.dumps(r.estimate, sort_keys=True),
                        charged="{}", usage_source="none", estimated_cost_microusd=r.estimated_cost_microusd,
                        cost_source="none", created_at=now, reserved_until_ms=_ms(r.reserved_until)))
                    s.flush()
                return
            except IntegrityError:
                with self._sessions() as s:
                    if s.get(SpendEntryRow, r.entry_id) is not None:
                        return  # this attempt was already reserved: reserving it again changes nothing
                    if r.idempotency_key and s.scalar(select(SpendEntryRow.entry_id).where(
                            SpendEntryRow.idempotency_key == r.idempotency_key,
                            SpendEntryRow.state == SpendState.RESERVED.value)) is not None:
                        raise SpendInFlight(r.idempotency_key) from None
                raise
            except DBAPIError as exc:
                if attempt < self.TRIES and _retryable(exc):
                    continue
                raise

    def settle(self, entry_id: str, settlement: SpendSettlement, *, now: datetime) -> SettleResult:
        st = settlement
        sources = [SpendState.RESERVED] + ([SpendState.UNKNOWN] if st.state == SpendState.SETTLED else [])
        values = {"state": st.state.value, "charged": json.dumps(st.charged, sort_keys=True),
                  "reported": json.dumps(st.reported, sort_keys=True) if st.reported is not None else None,
                  "usage_source": st.usage_source, "final_cost_microusd": st.final_cost_microusd,
                  "cost_source": st.cost_source, "currency": st.currency, "vendor_request_id": st.vendor_request_id,
                  "error_type": st.error_type, "latency_ms": st.latency_ms, "settled_at": now}
        for attempt in range(1, self.TRIES + 1):
            try:
                with self._sessions.begin() as s:
                    previous = None
                    for source in sources:
                        done = s.execute(update(SpendEntryRow).where(
                            SpendEntryRow.entry_id == entry_id, SpendEntryRow.state == source.value).values(**values))
                        if done.rowcount == 1:
                            previous = source
                            break
                    row = s.get(SpendEntryRow, entry_id)
                    if row is None:
                        raise NotFound(entry_id)
                    keys = json.loads(row.scope_keys)
                    if previous is None:
                        same = row.state == st.state.value and json.loads(row.charged or "{}") == st.charged
                        return SettleResult(outcome="duplicate" if same else "conflict", totals=self._totals(s, keys))
                    estimate = json.loads(row.estimate)
                    for key in sorted(keys):
                        for metric in sorted(set(estimate) | set(st.charged)):
                            if previous == SpendState.RESERVED:
                                d_reserved, d_settled = -estimate.get(metric, 0), st.charged.get(metric, 0)
                            else:  # unknown -> settled: the charged estimate is replaced by the real usage
                                d_reserved, d_settled = 0, st.charged.get(metric, 0) - estimate.get(metric, 0)
                            if d_reserved or d_settled:
                                _ensure_counter(s, key, metric, now)
                                s.execute(update(SpendCounterRow).where(
                                    SpendCounterRow.scope_key == key, SpendCounterRow.metric == metric).values(
                                    reserved=SpendCounterRow.reserved + d_reserved,
                                    settled=SpendCounterRow.settled + d_settled, updated_at=now))
                    return SettleResult(outcome="applied", totals=self._totals(s, keys),
                                        stale_lease=self._stale(s, row, now))
            except DBAPIError as exc:
                if attempt < self.TRIES and _retryable(exc):
                    continue
                raise
        raise AssertionError("unreachable")

    def expire_abandoned(self, now: datetime) -> list[str]:
        with self._sessions() as s:
            rows = s.scalars(select(SpendEntryRow).where(
                SpendEntryRow.state == SpendState.RESERVED.value, SpendEntryRow.reserved_until_ms < _ms(now))
                .order_by(SpendEntryRow.reserved_until_ms).limit(100)).all()
            entries = [self._entry(r) for r in rows]
        swept = []
        for entry in entries:  # one transaction each: an entry that cannot be settled does not stop the others
            try:
                if self.settle(entry.entry_id, abandoned(entry), now=now).outcome == "applied":
                    swept.append(entry.entry_id)
            except Exception:  # noqa: BLE001 - best-effort: the entry stays reserved for the next sweep
                logger.warning("abandoned spend entry %s could not be settled; the next sweep retries it",
                               entry.entry_id, exc_info=True)
        return swept

    def entries(self, *, task_id: str | None = None, session_id: str | None = None,
                attempt_ref: str | None = None) -> list[SpendEntry]:
        q = select(SpendEntryRow)
        if task_id is not None:
            q = q.where(SpendEntryRow.task_id == task_id)
        if session_id is not None:
            q = q.where(SpendEntryRow.session_id == session_id)
        if attempt_ref is not None:
            q = q.where(SpendEntryRow.attempt_ref == attempt_ref)
        with self._sessions() as s:
            return [self._entry(r) for r in s.scalars(q.order_by(SpendEntryRow.created_at, SpendEntryRow.operation_id,
                                                                 SpendEntryRow.attempt))]

    def totals(self, scope_keys: Iterable[str]) -> dict[str, dict[str, CounterTotals]]:
        with self._sessions() as s:
            return self._totals(s, scope_keys)

    def seed(self, scope_key: str, settled: dict[str, int], *, now: datetime) -> bool:
        with self._sessions.begin() as s:
            # Start with a write (SQLite then holds the write lock: concurrent seeders serialize). On PostgreSQL two
            # seeders may both find nothing; they insert the same values with ON CONFLICT DO NOTHING, never twice.
            s.execute(update(SpendCounterRow).where(SpendCounterRow.scope_key == scope_key)
                      .values(updated_at=SpendCounterRow.updated_at))
            if s.scalar(select(func.count()).select_from(SpendCounterRow).where(
                    SpendCounterRow.scope_key == scope_key)):
                return False
            for metric, amount in sorted(settled.items()):
                _ensure_counter(s, scope_key, metric, now, settled=amount)
            return True

    @staticmethod
    def _totals(s: Session, scope_keys: Iterable[str]) -> dict[str, dict[str, CounterTotals]]:
        out: dict[str, dict[str, CounterTotals]] = {}
        keys = list(scope_keys)
        if not keys:
            return out
        for row in s.scalars(select(SpendCounterRow).where(SpendCounterRow.scope_key.in_(keys))):
            out.setdefault(row.scope_key, {})[row.metric] = CounterTotals(reserved=row.reserved, settled=row.settled)
        return out

    @staticmethod
    def _stale(s: Session, row: SpendEntryRow, now: datetime) -> bool:
        """Whether the entry was reserved under a task lease that is no longer current (diagnostics only)."""
        if row.lease_fence is None or row.task_id is None:
            return False
        task = s.get(TaskRow, row.task_id)
        return task is None or not (task.lease_fence == row.lease_fence and task.lease_owner == row.worker
                                    and task.lease_expires_ms is not None and task.lease_expires_ms > _ms(now))

    @staticmethod
    def _entry(row: SpendEntryRow) -> SpendEntry:
        return SpendEntry(
            entry_id=row.entry_id, operation_id=row.operation_id, attempt=row.attempt,
            idempotency_key=row.idempotency_key, capability=row.capability, operation=row.operation,
            provider=row.provider, model=row.model, task_id=row.task_id, session_id=row.session_id,
            attempt_ref=row.attempt_ref, learner_id=row.learner_id, node_id=row.node_id, agent_id=row.agent_id,
            worker=row.worker, lease_fence=row.lease_fence, scope_keys=json.loads(row.scope_keys),
            state=SpendState(row.state), estimate=json.loads(row.estimate), charged=json.loads(row.charged or "{}"),
            reported=json.loads(row.reported) if row.reported else None, usage_source=row.usage_source,
            estimated_cost_microusd=row.estimated_cost_microusd, final_cost_microusd=row.final_cost_microusd,
            cost_source=row.cost_source, currency=row.currency, vendor_request_id=row.vendor_request_id,
            error_type=row.error_type, latency_ms=row.latency_ms, created_at=row.created_at,
            settled_at=row.settled_at,
            reserved_until=datetime.fromtimestamp(row.reserved_until_ms / 1000, tz=timezone.utc))


_ACTIVE = [s.value for s in ACTIVE_JOB_STATUSES]


def _at(ms: int | None) -> datetime | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class SqlTaskJobRepository:
    """The task job queue. Persistence only: the worker (app/services/worker.py) decides what a job does.

    Every write is one conditional statement in its own transaction: enqueue is an insert guarded by the partial
    unique index on the task's active job; a claim is an UPDATE that matches only a QUEUED, available job or a RUNNING
    one whose lease ran out; renew, finish and retry match only the claiming worker's own RUNNING job. Two workers can
    therefore never both claim a job, and a worker whose job was reclaimed cannot overwrite the new claim's outcome.
    """

    CANDIDATES = 8

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow) -> None:
        self._sessions = sessions
        self._clock = clock

    def enqueue(self, job: TaskJob) -> tuple[TaskJob, bool]:
        """Insert `job` unless its task already has an active job. Returns (the task's active job, inserted)."""
        existing = self.active_for_task(job.task_id)
        if existing is not None:
            return existing, False
        try:
            with self._sessions.begin() as s:
                s.add(self._row(job))
        except IntegrityError:  # a concurrent enqueue for the same task won
            existing = self.active_for_task(job.task_id)
            if existing is None:  # it already finished: this one is not a duplicate any more
                return self.enqueue(job)
            return existing, False
        return job, True

    def get(self, job_id: str) -> TaskJob:
        with self._sessions() as s:
            row = s.get(TaskJobRow, job_id)
            if row is None:
                raise NotFound(job_id)
            return self._job(row)

    def active_for_task(self, task_id: str) -> TaskJob | None:
        with self._sessions() as s:
            row = s.scalars(select(TaskJobRow).where(TaskJobRow.task_id == task_id,
                                                     TaskJobRow.status.in_(_ACTIVE))).first()
            return self._job(row) if row is not None else None

    def latest_for_task(self, task_id: str) -> TaskJob | None:
        with self._sessions() as s:
            row = s.scalars(select(TaskJobRow).where(TaskJobRow.task_id == task_id)
                            .order_by(TaskJobRow.created_at.desc(), TaskJobRow.job_id.desc())).first()
            return self._job(row) if row is not None else None

    def for_task(self, task_id: str) -> list[TaskJob]:
        with self._sessions() as s:
            rows = s.scalars(select(TaskJobRow).where(TaskJobRow.task_id == task_id)
                             .order_by(TaskJobRow.created_at, TaskJobRow.job_id))
            return [self._job(r) for r in rows]

    def list(self, status: JobStatus | None = None) -> list[TaskJob]:
        with self._sessions() as s:
            q = select(TaskJobRow).order_by(TaskJobRow.created_at, TaskJobRow.job_id)
            if status is not None:
                q = q.where(TaskJobRow.status == status.value)
            return [self._job(r) for r in s.scalars(q)]

    def _abandoned(self, now_ms: int):
        return (TaskJobRow.status == JobStatus.RUNNING.value) & (TaskJobRow.lease_expires_ms <= now_ms)

    def _claimable(self, now_ms: int):
        return or_(
            (TaskJobRow.status == JobStatus.QUEUED.value) & (TaskJobRow.available_at_ms <= now_ms),
            self._abandoned(now_ms) & (TaskJobRow.attempts < TaskJobRow.max_attempts))

    def fail_exhausted(self, now: datetime | None = None) -> int:
        """FAIL every job whose worker died (its lease ran out) on its last allowed attempt: a job that kills its
        worker each time is not claimed forever. One conditional UPDATE, so a racing claim cannot also take it.
        Returns how many jobs it failed."""
        now = now or self._clock()
        with self._sessions.begin() as s:
            done = s.execute(update(TaskJobRow).where(
                self._abandoned(_ms(now)), TaskJobRow.attempts >= TaskJobRow.max_attempts).values(
                status=JobStatus.FAILED.value, lease_owner=None, lease_expires_ms=None,
                error="the worker stopped (died or hung) on the job's last attempt",
                error_category=JobErrorCategory.WORKER_ERROR.value, updated_at=now, finished_at=now))
        return done.rowcount

    def claim(self, owner: str, ttl: timedelta) -> TaskJob | None:
        """Claim the oldest available job: a QUEUED one whose backoff has passed, or a RUNNING one whose worker's
        lease ran out (that worker died or hung; the claim counts as a new attempt). A RUNNING job whose worker died
        on its last allowed attempt is FAILED instead (`fail_exhausted`). None when there is none."""
        now = self._clock()
        now_ms = _ms(now)
        self.fail_exhausted(now)
        with self._sessions() as s:
            candidates = list(s.scalars(select(TaskJobRow.job_id).where(self._claimable(now_ms))
                                        .order_by(TaskJobRow.available_at_ms, TaskJobRow.created_at)
                                        .limit(self.CANDIDATES)))
        for job_id in candidates:
            with self._sessions.begin() as s:
                done = s.execute(update(TaskJobRow).where(TaskJobRow.job_id == job_id, self._claimable(now_ms))
                                 .values(status=JobStatus.RUNNING.value, lease_owner=owner,
                                         lease_expires_ms=_ms(now + ttl), attempts=TaskJobRow.attempts + 1,
                                         updated_at=now))
            if done.rowcount == 1:
                return self.get(job_id)
        return None

    def _owned(self, job_id: str, owner: str):
        return ((TaskJobRow.job_id == job_id) & (TaskJobRow.status == JobStatus.RUNNING.value)
                & (TaskJobRow.lease_owner == owner))

    def renew(self, job_id: str, owner: str, ttl: timedelta) -> bool:
        """Extend the worker's claim; False once the job was reclaimed by another worker (or finished)."""
        now = self._clock()
        with self._sessions.begin() as s:
            done = s.execute(update(TaskJobRow).where(self._owned(job_id, owner), TaskJobRow.lease_expires_ms > _ms(now))
                             .values(lease_expires_ms=_ms(now + ttl), updated_at=now))
        return done.rowcount == 1

    def finish(self, job_id: str, owner: str, status: JobStatus, *, task_status: TaskStatus | None = None,
               error: str | None = None, error_category: JobErrorCategory | None = None) -> bool:
        """Record the job's outcome (DONE, FAILED or CANCELLED); False when the job is no longer this worker's."""
        now = self._clock()
        with self._sessions.begin() as s:
            done = s.execute(update(TaskJobRow).where(self._owned(job_id, owner)).values(
                status=status.value, lease_owner=None, lease_expires_ms=None,
                task_status=task_status.value if task_status is not None else None, error=error,
                error_category=error_category.value if error_category is not None else None, updated_at=now,
                finished_at=now))
        return done.rowcount == 1

    def retry(self, job_id: str, owner: str, *, delay: timedelta, error: str | None = None,
              count_attempt: bool = True) -> bool:
        """Put the job back in the queue, not claimable for `delay`. `count_attempt=False` gives the attempt back
        (the task was busy: nothing ran). False when the job is no longer this worker's."""
        now = self._clock()
        values: dict = {"status": JobStatus.QUEUED.value, "lease_owner": None, "lease_expires_ms": None,
                        "available_at_ms": _ms(now + delay), "updated_at": now}
        if error is not None:
            values["error"] = error
        if not count_attempt:
            values["attempts"] = TaskJobRow.attempts - 1
        with self._sessions.begin() as s:
            done = s.execute(update(TaskJobRow).where(self._owned(job_id, owner)).values(**values))
        return done.rowcount == 1

    def cancel_queued(self, task_id: str) -> list[str]:
        """Cancel the task's QUEUED job (a cancelled task needs no more work). A RUNNING job is left to its worker,
        which stops at the task's next node boundary."""
        now = self._clock()
        with self._sessions() as s:
            ids = list(s.scalars(select(TaskJobRow.job_id).where(TaskJobRow.task_id == task_id,
                                                                 TaskJobRow.status == JobStatus.QUEUED.value)))
        cancelled = []
        for job_id in ids:
            with self._sessions.begin() as s:
                done = s.execute(update(TaskJobRow).where(TaskJobRow.job_id == job_id,
                                                          TaskJobRow.status == JobStatus.QUEUED.value)
                                 .values(status=JobStatus.CANCELLED.value, updated_at=now, finished_at=now))
            if done.rowcount == 1:
                cancelled.append(job_id)
        return cancelled

    @staticmethod
    def _row(job: TaskJob) -> TaskJobRow:
        return TaskJobRow(
            job_id=job.job_id, task_id=job.task_id, kind=job.kind.value, status=job.status.value,
            dedupe_key=job.dedupe_key, payload=json.dumps(job.payload), attempts=job.attempts,
            max_attempts=job.max_attempts, available_at_ms=_ms(job.available_at), lease_owner=job.lease_owner,
            lease_expires_ms=_ms(job.lease_until) if job.lease_until else None,
            task_status=job.task_status, error=job.error,
            error_category=job.error_category.value if job.error_category else None, created_at=job.created_at,
            updated_at=job.updated_at, finished_at=job.finished_at)

    @staticmethod
    def _job(row: TaskJobRow) -> TaskJob:
        return TaskJob(
            job_id=row.job_id, task_id=row.task_id, kind=JobKind(row.kind), status=JobStatus(row.status),
            payload=json.loads(row.payload), dedupe_key=row.dedupe_key, attempts=row.attempts,
            max_attempts=row.max_attempts, available_at=_at(row.available_at_ms), lease_owner=row.lease_owner,
            lease_until=_at(row.lease_expires_ms), task_status=row.task_status,
            error=row.error, error_category=JobErrorCategory(row.error_category) if row.error_category else None,
            created_at=_aware(row.created_at), updated_at=_aware(row.updated_at),
            finished_at=_aware(row.finished_at) if row.finished_at else None)


def _aware(at: datetime) -> datetime:
    """SQLite returns naive datetimes for timezone-aware columns: they were written in UTC."""
    return at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)


class SqlClassRepository:
    """Classes and their rosters. A join reads the class by its code under a row lock and inserts the membership in
    the same transaction, so a join never lands after a rotation that retired the code it used (PostgreSQL: the
    rotation's UPDATE waits for the lock; SQLite serializes writers). The roster's primary key makes a join
    idempotent, also when two joins race."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create(self, classroom: Classroom, join_code_hash: str) -> None:
        with self._sessions.begin() as s:
            s.add(ClassRow(class_id=classroom.class_id, teacher_id=classroom.teacher_id, tenant_id=classroom.tenant_id,
                           name=classroom.name, join_code_hash=join_code_hash, created_at=classroom.created_at))

    def get(self, class_id: str) -> Classroom | None:
        with self._sessions() as s:
            row = s.get(ClassRow, class_id)
            return self._class(row) if row is not None else None

    def for_teacher(self, teacher_id: str, tenant_id: str | None) -> list[Classroom]:
        with self._sessions() as s:
            rows = s.scalars(select(ClassRow).where(ClassRow.teacher_id == teacher_id, _tenant_is(tenant_id))
                             .order_by(ClassRow.created_at, ClassRow.class_id))
            return [self._class(r) for r in rows]

    def set_join_code(self, class_id: str, join_code_hash: str) -> bool:
        with self._sessions.begin() as s:
            done = s.execute(update(ClassRow).where(ClassRow.class_id == class_id)
                             .values(join_code_hash=join_code_hash))
            return done.rowcount == 1

    def join(self, join_code_hash: str, learner_id: str, at: datetime,
             tenant_id: str | None) -> tuple[Classroom, ClassMember, bool] | None:
        """Put the learner on the roster of the class in this tenant whose current code hashes to `join_code_hash`.
        None when there is no such class; otherwise the class, the membership and whether this call created it."""
        by_code = select(ClassRow).where(ClassRow.join_code_hash == join_code_hash, _tenant_is(tenant_id))
        try:
            with self._sessions.begin() as s:
                row = s.scalars(by_code.with_for_update()).first()
                if row is None:
                    return None
                classroom = self._class(row)
                existing = s.get(ClassMemberRow, (row.class_id, learner_id))
                if existing is not None:
                    return classroom, self._member(existing), False
                s.add(ClassMemberRow(class_id=row.class_id, learner_id=learner_id, joined_at=at))
                s.flush()
                return classroom, ClassMember(learner_id=learner_id, joined_at=at), True
        except IntegrityError:  # a concurrent join of the same learner committed first: theirs is the membership
            with self._sessions() as s:
                found = s.scalars(by_code).first()
                member = s.get(ClassMemberRow, (found.class_id, learner_id)) if found is not None else None
                if found is None or member is None:
                    raise
                return self._class(found), self._member(member), False

    def remove(self, class_id: str, learner_id: str) -> bool:
        with self._sessions.begin() as s:
            done = s.execute(delete(ClassMemberRow).where(ClassMemberRow.class_id == class_id,
                                                          ClassMemberRow.learner_id == learner_id))
            return done.rowcount == 1

    def members(self, class_id: str) -> list[ClassMember]:
        with self._sessions() as s:
            rows = s.scalars(select(ClassMemberRow).where(ClassMemberRow.class_id == class_id)
                             .order_by(ClassMemberRow.joined_at, ClassMemberRow.learner_id))
            return [self._member(r) for r in rows]

    def member(self, class_id: str, learner_id: str) -> ClassMember | None:
        with self._sessions() as s:
            row = s.get(ClassMemberRow, (class_id, learner_id))
            return self._member(row) if row is not None else None

    def enrollments(self, learner_id: str) -> list[tuple[Classroom, ClassMember]]:
        with self._sessions() as s:
            rows = s.execute(select(ClassRow, ClassMemberRow).join(ClassMemberRow,
                                                                   ClassMemberRow.class_id == ClassRow.class_id)
                             .where(ClassMemberRow.learner_id == learner_id)
                             .order_by(ClassMemberRow.joined_at, ClassRow.class_id))
            return [(self._class(c), self._member(m)) for c, m in rows]

    def teaches(self, teacher_id: str, tenant_id: str | None, learner_id: str) -> bool:
        """Whether `learner_id` is on the roster of a class of this teacher, in this tenant."""
        with self._sessions() as s:
            found = s.execute(select(ClassMemberRow.class_id).join(ClassRow,
                                                                   ClassRow.class_id == ClassMemberRow.class_id)
                              .where(ClassMemberRow.learner_id == learner_id, ClassRow.teacher_id == teacher_id,
                                     _tenant_is(tenant_id)).limit(1)).first()
            return found is not None

    @staticmethod
    def _class(row: ClassRow) -> Classroom:
        return Classroom(class_id=row.class_id, name=row.name, teacher_id=row.teacher_id, tenant_id=row.tenant_id,
                         created_at=_aware(row.created_at))

    @staticmethod
    def _member(row: ClassMemberRow) -> ClassMember:
        return ClassMember(learner_id=row.learner_id, joined_at=_aware(row.joined_at))


def _tenant_is(tenant_id: str | None):
    return ClassRow.tenant_id.is_(None) if tenant_id is None else ClassRow.tenant_id == tenant_id
