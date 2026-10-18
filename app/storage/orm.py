"""Metadata tables. Each row stores indexed columns plus the full schema as JSON."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TaskRow(Base):
    """A task. `version` is its optimistic lock (raised by every write of the body). The lease columns say which
    worker may execute it: `lease_fence` only ever grows (each claim raises it), and a holder's writes apply only while
    it is unchanged and `lease_expires_ms` is in the future. Pause and cancel requests are their own columns, so a
    request from another caller is never lost to the holder's next write."""

    __tablename__ = "tasks"
    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    lease_fence: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    lease_expires_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True, default=None)
    pause_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")


class EventRow(Base):
    __tablename__ = "events"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[str] = mapped_column(Text)


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("task_id", "name", "version", name="uq_artifact_version"),)
    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    type: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    body: Mapped[str] = mapped_column(Text)


class LearnerRow(Base):
    __tablename__ = "learners"
    learner_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[str] = mapped_column(Text)


class LearningEvidenceRow(Base):
    """Append-only: one immutable LearningEvidence per row, read back in recording order (seq)."""

    __tablename__ = "learning_evidence"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_id: Mapped[str] = mapped_column(String(64), unique=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    concept_id: Mapped[str] = mapped_column(String(128), index=True)
    source_type: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[str] = mapped_column(Text)


class LearningEventRow(Base):
    __tablename__ = "learning_events"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[str] = mapped_column(Text)


class LearningGoalRow(Base):
    __tablename__ = "learning_goals"
    goal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    domain: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32))
    body: Mapped[str] = mapped_column(Text)


class CurriculumRow(Base):
    """The current-version pointer of a goal's curriculum."""

    __tablename__ = "curricula"
    curriculum_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    goal_id: Mapped[str] = mapped_column(String(64), index=True)
    body: Mapped[str] = mapped_column(Text)


class CurriculumVersionRow(Base):
    """Append-only: one immutable curriculum version per row."""

    __tablename__ = "curriculum_versions"
    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    curriculum_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    body: Mapped[str] = mapped_column(Text)


class CurriculumProgressRow(Base):
    """The last progress snapshot of a curriculum (to report objective transitions once)."""

    __tablename__ = "curriculum_progress"
    curriculum_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    body: Mapped[str] = mapped_column(Text)


class TeachingSessionRow(Base):
    """An interactive teaching session; `version` is its optimistic lock."""

    __tablename__ = "teaching_sessions"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    lesson_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    body: Mapped[str] = mapped_column(Text)


class TeachingTurnRow(Base):
    """Append-only: one turn per row; a sequence number is stored once per session."""

    __tablename__ = "teaching_turns"
    __table_args__ = (UniqueConstraint("session_id", "sequence"),)
    turn_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    body: Mapped[str] = mapped_column(Text)


class InteractionEvidenceRow(Base):
    """Append-only: one interaction evidence item per row, read back in recording order (seq)."""

    __tablename__ = "interaction_evidence"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_id: Mapped[str] = mapped_column(String(64), unique=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    body: Mapped[str] = mapped_column(Text)


class TeachingRequestRow(Base):
    """An applied learner request by its client_turn_id (idempotent answers)."""

    __tablename__ = "teaching_requests"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_turn_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    body: Mapped[str] = mapped_column(Text)


class TeachingOutboxRow(Base):
    """Events and artifacts to publish after a session change, stored in the change's transaction."""

    __tablename__ = "teaching_outbox"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[str] = mapped_column(String(64), unique=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    body: Mapped[str] = mapped_column(Text)


class AssessmentItemRow(Base):
    """An immutable assessment item (the full item as JSON; its content hash detects a conflicting re-registration)."""

    __tablename__ = "assessment_items"
    item_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    lesson_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    body: Mapped[str] = mapped_column(Text)


class AssessmentRubricRow(Base):
    __tablename__ = "assessment_rubrics"
    rubric_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    body: Mapped[str] = mapped_column(Text)


class AssessmentAttemptRow(Base):
    """Append-only: one attempt per row. The attempt number is unique per learner and item."""

    __tablename__ = "assessment_attempts"
    __table_args__ = (UniqueConstraint("item_id", "learner_id", "attempt_number"),)
    attempt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    item_id: Mapped[str] = mapped_column(String(128), index=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    grade_id: Mapped[str] = mapped_column(String(64), unique=True)
    body: Mapped[str] = mapped_column(Text)


class AssessmentGradeRow(Base):
    """Append-only and immutable: one grade per attempt."""

    __tablename__ = "assessment_grades"
    grade_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(String(128), unique=True)
    body: Mapped[str] = mapped_column(Text)


class LearningCycleRow(Base):
    """A learning cycle; `version` is its optimistic lock."""

    __tablename__ = "learning_cycles"
    cycle_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    learner_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[str] = mapped_column(Text)
    # Copies of the body's drive lease, so a driver's write can check its token and expiry in the same statement.
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    lease_expires_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True, default=None)


class LearningCycleSlotRow(Base):
    """At most one active (running, waiting or blocked) cycle per learner: the row exists while it is active."""

    __tablename__ = "learning_cycle_slots"
    learner_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    cycle_id: Mapped[str] = mapped_column(String(64), unique=True)


class LearningCycleRequestRow(Base):
    """A learner response received by a cycle, by its client_response_id (idempotent responses)."""

    __tablename__ = "learning_cycle_requests"
    cycle_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_response_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    body: Mapped[str] = mapped_column(Text)


class SpendEntryRow(Base):
    """The spend ledger: one row per billable provider attempt, reserved before it is sent and settled once. Ids and
    numbers only (amounts in counter units: tokens, characters, ms, micro-USD)."""

    __tablename__ = "spend_entries"
    __table_args__ = (
        UniqueConstraint("operation_id", "attempt", name="uq_spend_operation_attempt"),
        Index("ix_spend_state_until", "state", "reserved_until_ms"),
        # At most one in-flight (reserved) attempt per idempotency key.
        Index("uq_spend_inflight_key", "idempotency_key", unique=True,
              sqlite_where=text("state = 'reserved' AND idempotency_key IS NOT NULL"),
              postgresql_where=text("state = 'reserved' AND idempotency_key IS NOT NULL")),
    )
    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # our request id (preq_...)
    operation_id: Mapped[str] = mapped_column(String(64))
    attempt: Mapped[int] = mapped_column(Integer)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    capability: Mapped[str] = mapped_column(String(32))
    operation: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    attempt_ref: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    learner_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    node_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    worker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_fence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scope_keys: Mapped[str] = mapped_column(Text)  # JSON list
    state: Mapped[str] = mapped_column(String(16))
    estimate: Mapped[str] = mapped_column(Text)  # JSON {metric: amount}
    charged: Mapped[str] = mapped_column(Text, default="{}")  # JSON {metric: amount}
    reported: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON provider-reported units
    usage_source: Mapped[str] = mapped_column(String(16), default="none")
    estimated_cost_microusd: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    final_cost_microusd: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cost_source: Mapped[str] = mapped_column(String(16), default="none")
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    vendor_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reserved_until_ms: Mapped[int] = mapped_column(BigInteger)


class SpendCounterRow(Base):
    """A budget scope's running totals for one metric: `reserved` (estimates of attempts in flight) and `settled`
    (what settled attempts charged). Kept in step with spend_entries in the same transactions."""

    __tablename__ = "spend_counters"
    scope_key: Mapped[str] = mapped_column(String(192), primary_key=True)
    metric: Mapped[str] = mapped_column(String(32), primary_key=True)
    reserved: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    settled: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
