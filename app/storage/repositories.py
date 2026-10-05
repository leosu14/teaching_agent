"""SQL repositories: persistence only, no business rules."""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

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
from app.schemas.curriculum import CurriculumProgress, CurriculumRecord, CurriculumVersion, VersionConflict
from app.schemas.events import Event
from app.schemas.learner import EvidenceConflict, LearnerProfile, LearningEvent, LearningEvidence, LearningGoal
from app.schemas.task import Task
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
    CurriculumProgressRow,
    CurriculumRow,
    CurriculumVersionRow,
    EventRow,
    LearnerRow,
    LearningEventRow,
    LearningEvidenceRow,
    LearningGoalRow,
    TaskRow,
    InteractionEvidenceRow,
    TeachingOutboxRow,
    TeachingRequestRow,
    TeachingSessionRow,
    TeachingTurnRow,
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
            if s.scalar(select(EventRow.seq).where(EventRow.event_id == event.event_id)) is not None:
                return  # an event with a stable id published again (e.g. after a restart) is stored once
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

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

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

    def save_rubric(self, rubric: AssessmentRubric) -> bool:
        return self._save(AssessmentRubricRow, rubric.rubric_id, rubric, rubric_id=rubric.rubric_id)

    def rubric(self, rubric_id: str) -> AssessmentRubric | None:
        with self._sessions() as s:
            row = s.get(AssessmentRubricRow, rubric_id)
            return AssessmentRubric.model_validate_json(row.body) if row else None

    def add_attempt(self, attempt: AssessmentAttempt, grade: AssessmentGrade) -> AssessmentAttempt:
        for _ in range(self.ATTEMPT_RETRIES):
            try:
                with self._sessions.begin() as s:
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

    def complete(self, attempt_id: str, outcome: AttemptOutcome | None, at) -> AssessmentAttempt:
        with self._sessions.begin() as s:
            row = s.get(AssessmentAttemptRow, attempt_id)
            if row is None:
                raise KeyError(attempt_id)
            current = AssessmentAttempt.model_validate_json(row.body)
            if current.completed_at is None:
                current = current.model_copy(update={"completed_at": at, "outcome": outcome})
                row.body = current.model_dump_json()
            return current
