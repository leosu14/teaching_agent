"""Learning cycles: the deterministic rules that execute one curriculum action end to end.

A cycle runs the NextLearningAction the curriculum selected, by composing what already exists: the lesson workflow,
the interactive teaching session (graded by the AssessmentService) and the evaluation workflow. This module is pure
(no I/O): the step policy, the guards and the state transitions. Every transition returns a new cycle with the next
version and the events it implies (stable ids), which the caller stores in one version-checked write and publishes
after it. `LearningCycleService` (services layer) drives the children and feeds their state back as observations.

    LEARN     LESSON (generated for the cycle)              -> TEACHING_SESSION (LEARN)
    REVIEW    LESSON (the objective's lesson, else generated) -> TEACHING_SESSION (REVIEW)
    PRACTICE  LESSON (the objective's lesson, else generated) -> TEACHING_SESSION (PRACTICE)
    EVALUATE  LESSON (the objective's lesson, else generated) -> EVALUATION
    WAIT      nothing: the cycle completes with NOTHING_DUE
    COMPLETE  nothing: the completion rule is verified, the cycle completes with GOAL_COMPLETED

Statuses: RUNNING -> (WAITING <-> RUNNING)* -> COMPLETED; RUNNING/WAITING -> BLOCKED (retryable failure) -> RUNNING
on resume; any active status -> FAILED (permanent) or CANCELLED.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.schemas.curriculum import LESSON_ACTIONS, LearningActionType, NextLearningAction, ObjectiveProgress
from app.schemas.curriculum import ObjectiveStatus
from app.schemas.events import EventType
from app.schemas.learner import stable_id
from app.schemas.learning_cycle import (
    ACTIVE_CYCLE_STATUSES,
    CycleConflict,
    CycleEvent,
    CycleFailure,
    CycleLease,
    CycleOutcome,
    CycleStatus,
    CycleStep,
    CycleWaiting,
    FailureKind,
    LearningCycle,
    StepKind,
    StepStatus,
    WaitKind,
)

__all__ = ["CycleConflict", "CycleNotFound", "InvalidCycleRequest", "InvalidCycleTransition", "Observation",
           "STEP_POLICY", "action_problem", "attach_child", "begin_step", "cancel", "classify_task_failure",
           "complete", "cycle_id_for", "fail", "observe", "plan_steps", "published", "receive", "record_artifact",
           "release", "resume", "claim", "start"]


class CycleNotFound(KeyError):
    """No such cycle (404)."""


class InvalidCycleTransition(ValueError):
    """The cycle's status does not allow the request: a response while nothing is asked, resuming a failed cycle (409)."""


class InvalidCycleRequest(ValueError):
    """The response does not fit the prompt: answers for a session question, a sheet with unknown questions (422)."""


# action -> steps: (kind, an existing lesson for the objective may serve, the session's action)
STEP_POLICY: dict[LearningActionType, tuple[tuple[StepKind, bool, str | None], ...]] = {
    LearningActionType.LEARN: ((StepKind.LESSON, False, None), (StepKind.TEACHING_SESSION, False, "LEARN")),
    LearningActionType.REVIEW: ((StepKind.LESSON, True, None), (StepKind.TEACHING_SESSION, False, "REVIEW")),
    LearningActionType.PRACTICE: ((StepKind.LESSON, True, None), (StepKind.TEACHING_SESSION, False, "PRACTICE")),
    LearningActionType.EVALUATE: ((StepKind.LESSON, True, None), (StepKind.EVALUATION, False, None)),
    LearningActionType.WAIT: (),
    LearningActionType.COMPLETE: (),
}


def cycle_id_for(learner_id: str, idempotency_key: str) -> str:
    """The same learner and key always name the same cycle."""
    return stable_id("lcyc", learner_id, idempotency_key)


def plan_steps(cycle_id: str, action: NextLearningAction) -> list[CycleStep]:
    if action.action not in STEP_POLICY:
        raise InvalidCycleRequest(f"no step policy for action {action.action}")
    return [CycleStep(step_id=f"{cycle_id}:{i}:{kind.value.lower()}", index=i, kind=kind,
                      child_key=stable_id("lcstep", cycle_id, str(i), kind.value), reuse_lesson=reuse,
                      session_action=session_action)  # type: ignore[arg-type]
            for i, (kind, reuse, session_action) in enumerate(STEP_POLICY[action.action])]


def action_problem(action: NextLearningAction, progress: ObjectiveProgress | None) -> str | None:
    """Why the action cannot be executed now (a deterministic validation failure), or None. The curriculum selected
    it, but the learner state may have moved since (another cycle, a direct session, an evaluation)."""
    if action.action not in LESSON_ACTIONS:
        return None
    if progress is None:
        return f"objective {action.objective_id} is not in the goal's current curriculum"
    if progress.status == ObjectiveStatus.BLOCKED:
        return f"objective {action.objective_id} is blocked by {', '.join(progress.unmet_prerequisites)}"
    if progress.status == ObjectiveStatus.MASTERED and action.action != LearningActionType.REVIEW:
        return f"objective {action.objective_id} is already mastered: a {action.action.value} action is stale"
    return None


def classify_task_failure(category: str | None) -> tuple[FailureKind, bool]:
    """A failed child task's category -> (kind, retryable). Provider failures and workflow stages are retried by the
    orchestrator's own recovery (it resumes a FAILED task from its checkpoint); a configuration error or an exhausted
    budget would fail the same way again."""
    if category == "ProviderError":
        return FailureKind.PROVIDER, True
    if category == "ConfigurationError":
        return FailureKind.VALIDATION, False
    if category == "BudgetExceededError":
        return FailureKind.WORKFLOW, False
    return FailureKind.WORKFLOW, True


@dataclass(frozen=True)
class Observation:
    """A child's state as the cycle needs it. `waiting` = (kind, ref) where the learner is needed."""

    state: str  # running | waiting | completed | failed | cancelled
    waiting: tuple[WaitKind, str] | None = None
    failure: CycleFailure | None = None


# --- transitions -------------------------------------------------------------------------------------------------


def start(*, learner_id: str, user_id: str, idempotency_key: str, action: NextLearningAction,
          at: datetime) -> LearningCycle:
    cycle_id = cycle_id_for(learner_id, idempotency_key)
    cycle = LearningCycle(cycle_id=cycle_id, learner_id=learner_id, user_id=user_id, idempotency_key=idempotency_key,
                          action=action, steps=plan_steps(cycle_id, action), created_at=at, updated_at=at)
    _event(cycle, EventType.LEARNING_CYCLE_STARTED, steps=[s.kind.value for s in cycle.steps])
    _event(cycle, EventType.LEARNING_CYCLE_ACTION_SELECTED, reason=action.reason,
           curriculum_version=action.curriculum_version,
           priority=action.priority.score if action.priority is not None else None)
    return cycle


def claim(cycle: LearningCycle, at: datetime, until: datetime) -> LearningCycle | None:
    """The drive lease, or None while another holder's lease runs. Written (version-checked) before any child is
    created, run, resumed or given a response, so two requests never drive the same cycle's children at once; a
    lease left by a crash expires at `until`. The token is the claiming version: unique, not random."""
    if cycle.lease is not None and cycle.lease.until > at:
        return None
    new = _next(cycle, at)
    new.lease = CycleLease(token=stable_id("lclease", cycle.cycle_id, str(new.version)), until=until)
    return new


def release(cycle: LearningCycle, token: str, at: datetime) -> LearningCycle | None:
    """Give the lease back (None when it is not held under `token`: it expired and was taken over)."""
    if cycle.lease is None or cycle.lease.token != token:
        return None
    new = _next(cycle, at)
    new.lease = None
    return new


def record_artifact(cycle: LearningCycle, key: str, artifact_id: str, at: datetime) -> LearningCycle | None:
    if cycle.artifact_ids.get(key) == artifact_id:
        return None
    new = _next(cycle, at)
    new.artifact_ids[key] = artifact_id
    return new


def begin_step(cycle: LearningCycle, step_id: str, at: datetime) -> LearningCycle:
    """Record the step's child key before the child exists: a crash after the child was created finds it by key."""
    _require_active(cycle)
    new = _next(cycle, at)
    step = new.step(step_id)
    if step.status != StepStatus.PENDING:
        raise InvalidCycleTransition(f"step {step_id} already started")
    step.status, step.started_at = StepStatus.STARTED, at
    _event(new, EventType.LEARNING_CYCLE_STEP_STARTED, step_id=step_id, step=step.kind.value)
    return new


def attach_child(cycle: LearningCycle, step_id: str, child_id: str, at: datetime, *,
                 reused: bool = False) -> LearningCycle:
    new = _next(cycle, at)
    step = new.step(step_id)
    if step.child_id is not None and step.child_id != child_id:
        raise CycleConflict(f"step {step_id} already has child {step.child_id}")
    step.child_id, step.reused, step.status = child_id, reused, StepStatus.RUNNING
    return new


def observe(cycle: LearningCycle, step_id: str, obs: Observation, at: datetime) -> LearningCycle | None:
    """Fold the child's state into the cycle. None when nothing changed (a read that observes the same state)."""
    step = cycle.step(step_id)
    if obs.state == "failed":
        assert obs.failure is not None
        return fail(cycle, obs.failure, at)
    if obs.state == "cancelled":
        return cancel(cycle, at, reason="the step's child was cancelled")
    if obs.state == "waiting":
        assert obs.waiting is not None
        kind, ref = obs.waiting
        if (cycle.status == CycleStatus.WAITING and cycle.waiting is not None and cycle.waiting.ref == ref
                and step.status == StepStatus.WAITING):
            return None
        new = _next(cycle, at)
        s = new.step(step_id)
        s.status = StepStatus.WAITING
        new.status, new.failure = CycleStatus.WAITING, None
        new.waiting = CycleWaiting(step_id=step_id, kind=kind, child_id=s.child_id or "", ref=ref, since=at)
        _event(new, EventType.LEARNING_CYCLE_WAITING, step_id=step_id, step=s.kind.value, wait=kind,
               child_id=s.child_id)
        return new
    if obs.state == "running":
        if step.status == StepStatus.RUNNING and cycle.status == CycleStatus.RUNNING:
            return None
        new = _next(cycle, at)
        new.step(step_id).status = StepStatus.RUNNING
        new.status, new.waiting, new.failure = CycleStatus.RUNNING, None, None
        return new
    assert obs.state == "completed", obs.state
    if step.status == StepStatus.COMPLETED:
        return None
    new = _next(cycle, at)
    s = new.step(step_id)
    s.status, s.completed_at = StepStatus.COMPLETED, at
    new.status, new.waiting, new.failure = CycleStatus.RUNNING, None, None
    _event(new, EventType.LEARNING_CYCLE_STEP_COMPLETED, step_id=step_id, step=s.kind.value, child_id=s.child_id,
           reused=s.reused)
    if new.current_step is None:
        _event(new, EventType.LEARNING_CYCLE_ACTION_COMPLETED, steps=len(new.steps))
    return new


def receive(cycle: LearningCycle, at: datetime) -> LearningCycle:
    """A learner response for the waiting point: the cycle runs again (the child gets the response next)."""
    if cycle.status != CycleStatus.WAITING or cycle.waiting is None:
        raise InvalidCycleTransition(f"cycle {cycle.cycle_id} is {cycle.status.value}: it is not waiting for the "
                                     f"learner")
    new = _next(cycle, at)
    w = new.waiting
    assert w is not None
    new.status = CycleStatus.RUNNING
    new.step(w.step_id).status = StepStatus.RUNNING
    _event(new, EventType.LEARNING_CYCLE_RESPONSE_RECEIVED, step_id=w.step_id, wait=w.kind)
    new.waiting = None
    return new


def complete(cycle: LearningCycle, outcome: CycleOutcome, at: datetime) -> LearningCycle:
    _require_active(cycle)
    if cycle.current_step is not None:
        raise InvalidCycleTransition(f"step {cycle.current_step.step_id} is not completed")
    new = _next(cycle, at)
    new.status, new.outcome, new.completed_at, new.waiting = CycleStatus.COMPLETED, outcome, at, None
    if not new.steps:
        _event(new, EventType.LEARNING_CYCLE_ACTION_COMPLETED, steps=0)
    next_action = outcome.next_action
    _event(new, EventType.LEARNING_CYCLE_COMPLETED, result=outcome.result.value,
           learning_evidence=len(outcome.learning_evidence_ids), mastery_updated=bool(outcome.mastery_changes),
           goal_complete=outcome.goal_complete,
           next_action=next_action.action.value if next_action else None,
           next_concept_id=next_action.concept_id if next_action else None)
    return new


def fail(cycle: LearningCycle, failure: CycleFailure, at: datetime) -> LearningCycle | None:
    """BLOCKED when retryable (the cycle keeps the learner's slot; `resume` retries), FAILED otherwise."""
    _require_active(cycle)
    if cycle.status == CycleStatus.BLOCKED and cycle.failure is not None and \
            (cycle.failure.kind, cycle.failure.step_id, cycle.failure.category) == \
            (failure.kind, failure.step_id, failure.category):
        return None  # the same blocked child observed again
    new = _next(cycle, at)
    new.status = CycleStatus.BLOCKED if failure.retryable else CycleStatus.FAILED
    new.failure, new.waiting = failure, None
    if failure.step_id is not None and not failure.retryable:
        new.step(failure.step_id).status = StepStatus.FAILED
    if new.status == CycleStatus.FAILED:
        new.completed_at = at
    _event(new, EventType.LEARNING_CYCLE_FAILED, kind=failure.kind.value, retryable=failure.retryable,
           category=failure.category, step_id=failure.step_id, status=new.status.value)
    return new


def resume(cycle: LearningCycle, at: datetime) -> LearningCycle | None:
    """BLOCKED -> RUNNING (the service then retries the child through its own recovery). Idempotent for an active or
    completed cycle; a permanently failed or cancelled cycle cannot be resumed."""
    if cycle.status in (CycleStatus.FAILED, CycleStatus.CANCELLED):
        raise InvalidCycleTransition(f"cycle {cycle.cycle_id} is {cycle.status.value} and cannot be resumed")
    if cycle.status != CycleStatus.BLOCKED:
        return None
    new = _next(cycle, at)
    failed = new.failure
    new.status, new.failure = CycleStatus.RUNNING, None
    _event(new, EventType.LEARNING_CYCLE_RESUMED, after=failed.kind.value if failed else None)
    return new


def cancel(cycle: LearningCycle, at: datetime, *, reason: str = "cancelled by request") -> LearningCycle | None:
    """Idempotent for a cancelled cycle; a completed or failed cycle is final."""
    if cycle.status == CycleStatus.CANCELLED:
        return None
    if cycle.status not in ACTIVE_CYCLE_STATUSES:
        raise InvalidCycleTransition(f"cycle {cycle.cycle_id} is {cycle.status.value}")
    new = _next(cycle, at)
    current = new.current_step
    if current is not None and current.status != StepStatus.PENDING:
        current.status = StepStatus.CANCELLED
    new.status, new.waiting, new.completed_at = CycleStatus.CANCELLED, None, at
    _event(new, EventType.LEARNING_CYCLE_CANCELLED, reason=reason)
    return new


def published(cycle: LearningCycle, event_ids: set[str], at: datetime) -> LearningCycle:
    """The events with these ids were published: drop them from the outbox."""
    new = _next(cycle, at)
    new.pending_events = [e for e in new.pending_events if e.event_id not in event_ids]
    return new


# --- helpers -----------------------------------------------------------------------------------------------------


def _require_active(cycle: LearningCycle) -> None:
    if cycle.status not in ACTIVE_CYCLE_STATUSES:
        raise InvalidCycleTransition(f"cycle {cycle.cycle_id} is {cycle.status.value}")


def _next(cycle: LearningCycle, at: datetime) -> LearningCycle:
    new = cycle.model_copy(deep=True)
    new.version += 1
    new.updated_at = at
    return new


def _event(cycle: LearningCycle, type: str, **data: object) -> None:
    """An event of this change, identified by the cycle, its type and the version that produced it: republishing it
    (after a crash) stores it once. No learner id, answer, prompt or provider detail."""
    a = cycle.action
    payload = {"cycle_id": cycle.cycle_id, "action": a.action.value, "goal_id": a.goal_id,
               "objective_id": a.objective_id, "concept_id": a.concept_id, **data}
    cycle.pending_events.append(CycleEvent(event_id=stable_id("lcevt", cycle.cycle_id, type, str(cycle.version)),
                                           type=type, data=payload))
