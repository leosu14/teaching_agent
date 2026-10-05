"""Learning cycles: the application service that executes the curriculum's next learning action end to end.

An orchestration layer, not a teacher. The existing systems stay authoritative:
- the action is selected by the curriculum's NextActionEngine (`CurriculumService.next_action`), never by the cycle;
- lessons are the lesson workflow (`CurriculumService.create_lesson`), sessions the interactive teaching layer
  (graded by the AssessmentService), evaluations the evaluation workflow; the cycle never generates or grades;
- mastery changes only through learner memory's MasteryUpdater, inside those children, from graded evidence;
- objective progress, goal completion and the next action come from the curriculum engine, inside those children;
  the cycle copies them into its outcome.

The cycle is durable and idempotent: its record (`app/curriculum/cycle.py` transitions, one version-checked write
each) names its steps and their children by deterministic keys, so a repeated request, a concurrent one or a restart
after a crash finds the same children instead of creating new ones. The record is a projection of its children:
reading a cycle folds in whatever they did since (a learner who answered through the session endpoint directly, a
crash after a child finished). When the learner is needed the cycle is WAITING and the request returns: nothing
waits in the process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from app.artifacts.service import ArtifactService
from app.curriculum import cycle as engine
from app.curriculum.cycle import CycleNotFound, InvalidCycleRequest, InvalidCycleTransition, Observation
from app.curriculum.repository import LearningCycleRepository
from app.learner.memory import LearnerMemoryService
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.runtime.orchestrator.orchestrator import InvalidInput
from app.runtime.tasks.state_machine import InvalidTransition
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.common import utcnow
from app.schemas.curriculum import LESSON_ACTIONS, LearningActionType, NextLearningAction
from app.schemas.events import Event
from app.schemas.learner import stable_id
from app.schemas.learning_cycle import (
    ACTIVE_CYCLE_STATUSES,
    CycleConflict,
    CycleFailure,
    CycleOutcome,
    CycleRequestRecord,
    CycleResponse,
    CycleResult,
    CycleStatus,
    CycleStep,
    FailureKind,
    LearnerPrompt,
    LearningCycle,
    LearningCycleView,
    StartLearningCycle,
    StepKind,
)
from app.schemas.task import Task, TaskStatus
from app.schemas.teaching import (
    LearnerInput,
    Speaker,
    StartTeachingSession,
    TeachingSession,
    TeachingSessionStatus,
)
from app.services.curriculum import CurriculumService, InvalidGoal
from app.services.tasks import TaskService
from app.services.teaching import (
    InvalidSessionTransition,
    InvalidTeachingRequest,
    TeacherUnavailable,
    TeachingSessionService,
)
from app.storage.repositories import SqlEventRepository

__all__ = ["CycleConflict", "CycleNotFound", "InvalidCycleRequest", "InvalidCycleTransition", "LearningCycleService"]

CYCLE_KEY, STEP_KEY = "learning_cycle_id", "learning_cycle_step"  # child task metadata: found again by these
NODE = "learning_cycle"
MAX_TRANSITIONS = 64  # per request; a cycle has at most a handful of steps
DRIVE_LEASE = timedelta(minutes=10)  # a crashed request's lease expires after this; then the cycle can be resumed
CONFLICT_RETRIES = 5
LEARNER_WAITS = {"diagnostic_answers", "assessment_answers"}  # task waits for the learner (others wait on work)
Session = TeachingSessionStatus


class LearningCycleService:
    def __init__(self, repository: LearningCycleRepository, *, curriculum: CurriculumService, tasks: TaskService,
                 teaching: TeachingSessionService, memory: LearnerMemoryService, artifacts: ArtifactService,
                 events: EventBus, event_log: SqlEventRepository, clock: Callable[[], datetime] = utcnow,
                 drive_lease: timedelta = DRIVE_LEASE) -> None:
        self._repo = repository
        self._curriculum = curriculum
        self._tasks = tasks
        self._teaching = teaching
        self._memory = memory
        self._artifacts = artifacts
        self._events = events
        self._event_log = event_log
        self._clock = clock
        self._lease = drive_lease

    @property
    def repository(self) -> LearningCycleRepository:
        return self._repo

    # --- start ---------------------------------------------------------------------------------------------------

    async def start(self, learner_id: str, data: StartLearningCycle) -> LearningCycleView:
        """Start the learner's next cycle and run it until the learner is needed or it ends. The same idempotency
        key returns the same cycle (continuing it, never selecting or executing a second action); another key while
        a cycle is active is a conflict."""
        self._memory.get(learner_id)  # unknown learners are rejected
        cycle_id = engine.cycle_id_for(learner_id, data.idempotency_key)
        created = False
        existing = self._repo.get(cycle_id)
        if existing is not None and existing.user_id != data.user_id:
            raise CycleConflict(f"idempotency_key {data.idempotency_key} was already used for a different request")
        if existing is None:
            active = self._repo.active_for(learner_id)
            if active is not None:
                raise CycleConflict(f"learner {learner_id} has an active learning cycle {active.cycle_id} "
                                    f"({active.status.value}): continue, resume or cancel it first")
            await self._ensure_curricula(learner_id, data.user_id)
            now = self._clock()
            action = await self._curriculum.next_action(learner_id, as_of=now)
            created = self._repo.create(engine.start(learner_id=learner_id, user_id=data.user_id,
                                                     idempotency_key=data.idempotency_key, action=action, at=now))
        return await self._drive(cycle_id, created=created)

    async def _ensure_curricula(self, learner_id: str, user_id: str) -> None:
        """An active goal without a curriculum gets one (the existing, idempotent planning task)."""
        for goal in self._curriculum.goals(learner_id):
            if goal.is_active and self._curriculum.curriculum(goal.goal_id) is None:
                await self._curriculum.build_curriculum(goal.goal_id, user_id=user_id)

    # --- learner responses ---------------------------------------------------------------------------------------

    async def respond(self, cycle_id: str, data: CycleResponse) -> LearningCycleView:
        """The learner's response to the current prompt, delivered to the waiting child once. Idempotent per
        client_response_id: the same id and response replay; the same id with another response is a conflict."""
        cycle = await self._advance(cycle_id, token=None)  # fold in what happened since the last request
        digest = _digest(data)
        record = self._repo.request(cycle_id, data.client_response_id)
        if record is not None:
            if record.request_hash != digest:
                raise CycleConflict(f"client_response_id {data.client_response_id} was already used for a different "
                                    f"response")
            stored = record

            async def redeliver(held: LearningCycle) -> None:
                if not stored.applied:  # received, but a crash came before the child got it
                    await self._deliver(held, stored, data)

            return await self._drive(cycle_id, replayed=True, before=redeliver)
        self._require_waiting(cycle)
        _check_shape(cycle, data)
        self._check_sheet(cycle, data)

        async def receive(held: LearningCycle) -> None:
            self._require_waiting(held)  # checked again under the lease: the waiting point may have moved
            _check_shape(held, data)
            self._check_sheet(held, data)
            w = held.waiting
            assert w is not None
            received = CycleRequestRecord(cycle_id=cycle_id, client_response_id=data.client_response_id,
                                          request_hash=digest, step_id=w.step_id, waiting_ref=w.ref,
                                          created_at=self._clock())
            change = engine.receive(held, self._clock())
            self._repo.apply(change, expected_version=held.version, request=received)  # the id is stored once
            await self._deliver(await self._publish(change), received, data)

        try:
            return await self._drive(cycle_id, before=receive, busy_error=True)
        except CycleConflict:
            stored = self._repo.request(cycle_id, data.client_response_id)
            if stored is not None and stored.request_hash == digest:  # the same response, sent twice at once
                return await self._view(await self._advance(cycle_id, token=None), replayed=True)
            raise

    @staticmethod
    def _require_waiting(cycle: LearningCycle) -> None:
        if cycle.status != CycleStatus.WAITING or cycle.waiting is None:
            raise InvalidCycleTransition(f"cycle {cycle.cycle_id} is {cycle.status.value}: it is not waiting for the "
                                         f"learner")

    async def _deliver(self, cycle: LearningCycle, record: CycleRequestRecord, data: CycleResponse) -> None:
        """Hand the response to the child at the waiting point it was given for, at most once."""
        step = cycle.step(record.step_id)
        child = step.child_id
        assert child is not None
        if step.kind == StepKind.TEACHING_SESSION:
            turn_id = stable_id("lcresp", cycle.cycle_id, record.client_response_id)
            session = await self._teaching.get(child)
            q = session.state.pending_question
            seen = self._teaching.repository.request(child, turn_id) is not None
            if seen or (session.status == Session.WAITING_FOR_LEARNER and q is not None and q.turn_id == record.waiting_ref):
                try:
                    await self._teaching.submit(child, LearnerInput(answer=data.answer or "", kind=data.kind,
                                                                    client_turn_id=turn_id))
                except TeacherUnavailable:
                    pass  # the answer is stored; the cycle sees the session owe a reply and blocks (provider)
                except (InvalidSessionTransition, InvalidTeachingRequest) as exc:
                    await self._refused(cycle.cycle_id, record, exc)
        else:
            task = self._tasks.get(child)
            if task.status == TaskStatus.WAITING and task.waiting is not None and \
                    task.waiting.node_id == record.waiting_ref:
                try:
                    await self._tasks.submit_answers(child, {"answers": [a.model_dump(mode="json")
                                                                         for a in data.answers or []]})
                except (InvalidInput, InvalidTransition) as exc:
                    await self._refused(cycle.cycle_id, record, exc)
        self._repo.mark_applied(cycle.cycle_id, record.client_response_id)

    async def _refused(self, cycle_id: str, record: CycleRequestRecord, exc: Exception) -> None:
        """The child rejected the response (invalid input): nothing changed; the id may be used again."""
        self._repo.forget_request(cycle_id, record.client_response_id)
        await self._advance(cycle_id, token=None)  # back to WAITING on the same point
        raise InvalidCycleRequest(str(exc)) from exc

    # --- control -------------------------------------------------------------------------------------------------

    async def resume(self, cycle_id: str) -> LearningCycleView:
        """Continue the cycle: a BLOCKED cycle retries its child through the child's own recovery (a failed task is
        resumed from its checkpoint, a session regenerates the reply it owes); an active one is reconciled and
        driven. A permanently failed or cancelled cycle cannot be resumed."""
        engine.resume(await self._load(cycle_id), self._clock())  # FAILED or CANCELLED: rejected before any claim

        async def unblock(held: LearningCycle) -> None:
            change = engine.resume(held, self._clock())
            if change is not None:
                held = await self._commit(held, change)
                step = held.current_step
                if step is not None and step.child_id is not None:
                    await self._recover(step)

        return await self._drive(cycle_id, before=unblock)

    async def _recover(self, step: CycleStep) -> None:
        assert step.child_id is not None
        try:
            if step.kind == StepKind.TEACHING_SESSION:
                session = await self._teaching.get(step.child_id)
                if session.status in (Session.PAUSED, Session.ACTIVE):
                    await self._teaching.resume(step.child_id)
            else:
                task = self._tasks.get(step.child_id)
                if task.status in (TaskStatus.FAILED, TaskStatus.PAUSED):
                    await self._tasks.resume(step.child_id)
        except (TeacherUnavailable, InvalidTransition, InvalidSessionTransition):
            pass  # still failing: the next observation blocks the cycle again, with the child's own reason

    async def cancel(self, cycle_id: str) -> LearningCycleView:
        """Stop the cycle and its running child (whose history is kept). Idempotent; evidence already recorded by a
        finished step stays."""
        cycle = await self._load(cycle_id)
        if engine.cancel(cycle, self._clock()) is None:
            return await self._view(cycle)
        step = cycle.current_step
        if step is not None and step.child_id is not None and not step.reused:
            if step.kind == StepKind.TEACHING_SESSION:
                await self._teaching.cancel(step.child_id)
            elif self._tasks.get(step.child_id).status not in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                                                                TaskStatus.CANCELLED):
                self._tasks.cancel(step.child_id)
        for _ in range(3):
            cycle = await self._load(cycle_id)
            change = engine.cancel(cycle, self._clock())
            if change is None:
                break
            try:
                cycle = await self._commit(cycle, change)
                break
            except CycleConflict:
                continue
        return await self._view(cycle)

    # --- reads ---------------------------------------------------------------------------------------------------

    async def get(self, cycle_id: str) -> LearningCycleView:
        """The cycle, reconciled with its children (no child is driven by a read)."""
        return await self._view(await self._advance(cycle_id, token=None))

    def cycle(self, cycle_id: str) -> LearningCycle:
        found = self._repo.get(cycle_id)
        if found is None:
            raise CycleNotFound(f"learning cycle {cycle_id} not found")
        return found

    async def for_learner(self, learner_id: str) -> list[LearningCycleView]:
        self._memory.get(learner_id)
        return [await self._view(c, with_prompt=False) for c in self._repo.for_learner(learner_id)]

    def events(self, cycle_id: str) -> list[Event]:
        self.cycle(cycle_id)
        return self._event_log.list_for_task(cycle_id)

    def artifacts(self, cycle_id: str) -> list[Artifact]:
        self.cycle(cycle_id)
        return self._artifacts.list_for_task(cycle_id)

    # --- the driver ----------------------------------------------------------------------------------------------

    async def _drive(self, cycle_id: str, *, before: Callable[[LearningCycle], Awaitable[None]] | None = None,
                     busy_error: bool = False, created: bool = False, replayed: bool = False) -> LearningCycleView:
        """Take the drive lease, run `before` (a response, an unblock) and drive the cycle until the learner is needed
        or it ends, then give the lease back. While another request holds the lease nothing is driven: the cycle is
        returned as it stands (with `busy_until`), or, with `busy_error`, a conflict. Every child a cycle touches is
        driven only under the lease, so retries and concurrent requests never run one twice; the lease lives in the
        cycle record, so a worker can hold it the same way."""
        cycle, token = await self._claim(cycle_id)
        if token is None:
            if busy_error and cycle.status in ACTIVE_CYCLE_STATUSES:
                raise CycleConflict(f"cycle {cycle_id} is being driven by another request: retry")
            if before is not None and cycle.status not in ACTIVE_CYCLE_STATUSES:
                await before(cycle)  # a final cycle: `before` rejects the request or only settles a stored response
            return await self._view(await self._advance(cycle_id, token=None), created=created, replayed=replayed)
        try:
            if before is not None:
                await before(cycle)
            await self._advance(cycle_id, token=token)
        except Exception:
            await self._release(cycle_id, token)
            raise
        return await self._view(await self._release(cycle_id, token), created=created, replayed=replayed)

    async def _claim(self, cycle_id: str) -> tuple[LearningCycle, str | None]:
        for _ in range(CONFLICT_RETRIES):
            cycle = await self._load(cycle_id)
            if cycle.status not in ACTIVE_CYCLE_STATUSES:
                return cycle, None
            now = self._clock()
            change = engine.claim(cycle, now, now + self._lease)
            if change is None:
                return cycle, None
            try:
                self._repo.apply(change, expected_version=cycle.version)
            except CycleConflict:
                continue
            assert change.lease is not None
            return await self._publish(change), change.lease.token
        return await self._load(cycle_id), None

    async def _release(self, cycle_id: str, token: str) -> LearningCycle:
        for _ in range(CONFLICT_RETRIES):
            cycle = await self._load(cycle_id)
            change = engine.release(cycle, token, self._clock())
            if change is None:
                return cycle
            try:
                return await self._commit(cycle, change)
            except CycleConflict:
                continue
        return await self._load(cycle_id)

    async def _advance(self, cycle_id: str, *, token: str | None) -> LearningCycle:
        """Fold the children's state into the cycle and, under the lease `token`, do the work it owes: record its
        artifact, start the next step's child, run or resume a child, finish. Without the lease it only observes.
        Returns when the learner is needed, the cycle ended or blocked, or (without the lease) when only driving could
        change it. A write that lost to a concurrent one is recomputed from the stored cycle: every step is derived
        from the record and the children, and children are found by key, so recomputing never repeats work."""
        cycle = await self._load(cycle_id)
        drive = token is not None
        for _ in range(MAX_TRANSITIONS):
            if drive and (cycle.lease is None or cycle.lease.token != token):
                drive = False  # the lease expired and was taken over: stop driving
            if cycle.status not in (CycleStatus.RUNNING, CycleStatus.WAITING):
                break
            try:
                cycle, more = await self._step(cycle, drive=drive)
            except CycleConflict:
                cycle = await self._load(cycle_id)
                continue
            if not more:
                break
        return cycle

    async def _step(self, cycle: LearningCycle, *, drive: bool) -> tuple[LearningCycle, bool]:
        """One transition of the driver; (cycle, False) when nothing more can happen in this request."""
        now = self._clock()
        if "cycle" not in cycle.artifact_ids:
            if not drive:
                return cycle, False
            stored = self._store_cycle_artifact(cycle)
            return await self._commit(cycle, engine.record_artifact(cycle, "cycle", stored.artifact_id, now)), True
        step = cycle.current_step
        if step is None:
            if not drive:
                return cycle, False
            return await self._finish(cycle), True
        if step.status.value == "PENDING":
            if not drive:
                return cycle, False
            problem = await self._action_problem(cycle) if step.index == 0 else None
            change = engine.fail(cycle, self._failure(FailureKind.VALIDATION, problem, step), now) if problem \
                else engine.begin_step(cycle, step.step_id, now)
            return await self._commit(cycle, change), True
        if step.child_id is None:
            if not drive:
                return cycle, False
            return await self._create_child(cycle, step), True
        obs = await self._observe(cycle, step, drive=drive)
        change = engine.observe(cycle, step.step_id, obs, self._clock())
        if change is None:
            return cycle, False
        return await self._commit(cycle, change), True

    async def _action_problem(self, cycle: LearningCycle) -> str | None:
        a = cycle.action
        if a.action not in LESSON_ACTIONS:
            return None
        assert a.goal_id and a.objective_id
        curriculum = self._curriculum.curriculum(a.goal_id)
        if curriculum is None:
            return f"goal {a.goal_id} has no curriculum"
        if curriculum.version != a.curriculum_version:
            return (f"the curriculum changed since the action was selected (version {a.curriculum_version} -> "
                    f"{curriculum.version}): start a new cycle")
        progress = next((p for p in curriculum.progress.objectives if p.objective_id == a.objective_id), None)
        return engine.action_problem(a, progress)

    async def _create_child(self, cycle: LearningCycle, step: CycleStep) -> LearningCycle:
        """Find the step's child by its key, else create it, and record it (only under the drive lease, so no
        concurrent request creates a second one)."""
        child_id, reused = self._find_child(cycle, step), False
        if child_id is None:
            try:
                child_id, reused = await self._new_child(cycle, step)
            except InvalidGoal as exc:  # the curriculum moved under the action
                return await self._commit(cycle, engine.fail(
                    cycle, self._failure(FailureKind.VALIDATION, str(exc), step), self._clock()))
        return await self._commit(cycle, engine.attach_child(cycle, step.step_id, child_id, self._clock(),
                                                             reused=reused))

    def _find_child(self, cycle: LearningCycle, step: CycleStep) -> str | None:
        if step.kind == StepKind.TEACHING_SESSION:
            found = [s for s in self._teaching.sessions(cycle.learner_id)
                     if s.metadata.get("idempotency_key") == step.child_key]
            return found[0].session_id if found else None
        found = [t for t in self._tasks.list_for_learner(cycle.learner_id) if t.metadata.get(STEP_KEY) == step.child_key]
        return found[0].task_id if found else None

    async def _new_child(self, cycle: LearningCycle, step: CycleStep) -> tuple[str, bool]:
        a = cycle.action
        keys = {CYCLE_KEY: cycle.cycle_id, STEP_KEY: step.child_key}
        if step.kind == StepKind.LESSON:
            assert a.goal_id and a.concept_id
            if step.reuse_lesson:
                existing = self._curriculum.lesson_for(cycle.learner_id, a.goal_id, a.concept_id)
                if existing is not None:
                    return existing.task_id, True
            task = await self._curriculum.create_lesson(a, user_id=cycle.user_id, metadata=keys,
                                                        cycle_artifact_id=cycle.artifact_ids["cycle"])
            return task.task_id, False
        lesson = self._lesson_task_id(cycle)
        if step.kind == StepKind.EVALUATION:
            return self._tasks.create_evaluation(lesson, user_id=cycle.user_id, metadata=keys).task_id, False
        try:
            started = await self._teaching.start(lesson, StartTeachingSession(idempotency_key=step.child_key,
                                                                               action=step.session_action))
            return started.session_id, False
        except TeacherUnavailable:  # created, but its opening failed: found by key, then observed (and blocked)
            found = self._find_child(cycle, step)
            if found is None:
                raise
            return found, False

    @staticmethod
    def _lesson_task_id(cycle: LearningCycle) -> str:
        lesson = next(s for s in cycle.steps if s.kind == StepKind.LESSON)
        assert lesson.child_id is not None
        return lesson.child_id

    async def _observe(self, cycle: LearningCycle, step: CycleStep, *, drive: bool) -> Observation:
        """The child's state; when `drive` (under the lease), a child that owes work (created, crashed mid-run, a reply
        owed) is run or resumed first."""
        assert step.child_id is not None
        if step.kind == StepKind.TEACHING_SESSION:
            session = await self._teaching.get(step.child_id)
            if session.learner_id != cycle.learner_id:
                return self._failed(FailureKind.VALIDATION, False, "the session belongs to another learner", step)
            if drive and (session.status == Session.ACTIVE
                          or (session.status == Session.COMPLETED and session.outcome is None)):
                try:
                    await self._teaching.resume(step.child_id)
                except TeacherUnavailable as exc:
                    return self._failed(FailureKind.PROVIDER, True, str(exc), step)
                session = await self._teaching.get(step.child_id)
            return self._session_observation(session, step)
        task = self._tasks.get(step.child_id)
        if task.learner_id != cycle.learner_id:
            return self._failed(FailureKind.VALIDATION, False, "the task belongs to another learner", step)
        owes_work = task.status in (TaskStatus.CREATED, TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.REVIEWING) \
            or (task.status == TaskStatus.WAITING and task.waiting is not None
                and task.waiting.kind not in LEARNER_WAITS)
        if drive and owes_work:
            task = await (self._tasks.run(task.task_id) if task.status == TaskStatus.CREATED
                          else self._tasks.resume(task.task_id))
        return self._task_observation(task, step)

    def _task_observation(self, task: Task, step: CycleStep) -> Observation:
        if task.status == TaskStatus.COMPLETED:
            return Observation("completed")
        if task.status == TaskStatus.WAITING and task.waiting is not None and task.waiting.kind in LEARNER_WAITS:
            return Observation("waiting", waiting=(task.waiting.kind, task.waiting.node_id))  # type: ignore[arg-type]
        if task.status == TaskStatus.FAILED:
            error = task.errors[-1] if task.errors else None
            kind, retryable = engine.classify_task_failure(error.category if error else None)
            return self._failed(kind, retryable, f"{step.kind.value.lower()} task failed"
                                + (f" at {error.node_id}" if error and error.node_id else ""), step,
                                category=error.category if error else None)
        if task.status == TaskStatus.PAUSED:
            return self._failed(FailureKind.WORKFLOW, True, f"the {step.kind.value.lower()} task is paused", step)
        if task.status == TaskStatus.CANCELLED:
            return Observation("cancelled")
        return Observation("running")

    def _session_observation(self, session: TeachingSession, step: CycleStep) -> Observation:
        if session.status == Session.WAITING_FOR_LEARNER and session.state.pending_question is not None:
            return Observation("waiting", waiting=("session_answer", session.state.pending_question.turn_id))
        if session.status == Session.COMPLETED and session.outcome is not None:
            return Observation("completed")
        if session.status == Session.PAUSED:
            return self._failed(FailureKind.WORKFLOW, True, "the teaching session is paused", step)
        if session.status == Session.FAILED:
            return self._failed(FailureKind.WORKFLOW, False, "the teaching session failed", step)
        if session.status == Session.CANCELLED:
            return Observation("cancelled")
        return Observation("running")  # ACTIVE: the teacher owes a reply (driven on the next drive)

    # --- completion ----------------------------------------------------------------------------------------------

    async def _finish(self, cycle: LearningCycle) -> LearningCycle:
        """All steps are done: copy what the action produced from the children, record the outcome artifact and
        complete. COMPLETE verifies the curriculum's completion rule; WAIT executes nothing."""
        a = cycle.action
        now = self._clock()
        if a.action == LearningActionType.WAIT:
            outcome = CycleOutcome(result=CycleResult.NOTHING_DUE, next_action=a)
        elif a.action == LearningActionType.COMPLETE:
            assert a.goal_id
            curriculum = self._curriculum.curriculum(a.goal_id)
            if curriculum is None or not curriculum.progress.goal_complete:
                return await self._commit(cycle, engine.fail(cycle, self._failure(
                    FailureKind.VALIDATION, f"goal {a.goal_id}'s completion rule does not hold", None), now))
            outcome = CycleOutcome(result=CycleResult.GOAL_COMPLETED, goal_complete=True,
                                   next_action=await self._curriculum.next_action(cycle.learner_id, as_of=a.as_of))
        else:
            outcome = await self._step_outcome(cycle)
        stored = self._store_outcome_artifact(cycle, outcome)
        outcome.artifact_ids["outcome"] = stored.artifact_id
        return await self._commit(cycle, engine.complete(cycle, outcome, now))

    async def _step_outcome(self, cycle: LearningCycle) -> CycleOutcome:
        a = cycle.action
        last = cycle.steps[-1]
        assert last.child_id is not None and a.goal_id and a.objective_id
        lesson_artifact = self._artifacts.find(self._lesson_task_id(cycle), "lesson")
        ids = {"lesson": lesson_artifact.artifact_id} if lesson_artifact else {}
        if last.kind == StepKind.TEACHING_SESSION:
            session = await self._teaching.get(last.child_id)
            o = session.outcome
            assert o is not None
            ids.update({k: v for k, v in o.artifact_ids.items() if k in ("session", "summary", "learning_evidence",
                                                                        "learning_action")})
            outcome = CycleOutcome(
                result=CycleResult.TAUGHT, learning_evidence_ids=o.learning_evidence_ids,
                mastery_changes=o.mastery_changes, objective_progress=o.objective_progress,
                next_action=NextLearningAction.model_validate(o.learning_action) if o.learning_action else None,
                artifact_ids=ids)
        else:
            task = self._tasks.get(last.child_id)
            assert task.result is not None
            prefix = f"{task.task_id}/"
            evidence = [e.evidence_id for e in self._memory.evidence(cycle.learner_id)
                        if e.source_type == "evaluation" and e.source_ref.startswith(prefix)]
            for name in ("learner_evaluation", "learning_action"):
                found = self._artifacts.find(task.task_id, name)
                if found is not None:
                    ids[name] = found.artifact_id
            outcome = CycleOutcome(
                result=CycleResult.EVALUATED, learning_evidence_ids=evidence,
                mastery_changes=[c.model_dump(mode="json") for c in task.result.mastery_changes],
                next_action=task.result.learning_action, artifact_ids=ids)
        curriculum = self._curriculum.curriculum(a.goal_id)
        if curriculum is not None:
            progress = next((p for p in curriculum.progress.objectives if p.objective_id == a.objective_id), None)
            if outcome.objective_progress is None and progress is not None:
                outcome.objective_progress = progress.model_dump(mode="json")
            outcome.goal_complete = curriculum.progress.goal_complete
        return outcome

    # --- artifacts -----------------------------------------------------------------------------------------------

    def _store_cycle_artifact(self, cycle: LearningCycle) -> Artifact:
        """LEARNING_CYCLE, at the start: derives from the action's LEARNING_OBJECTIVE artifact (or the curriculum
        version for COMPLETE); the lesson the cycle generates derives from it. Deterministic content, so storing it
        again after a crash reuses it."""
        a = cycle.action
        parents = []
        if a.goal_id:
            version = next((v for v in self._curriculum.versions(a.goal_id) if v.version == a.curriculum_version),
                           None)
            if version is not None:
                parent = version.artifact_ids.get(a.objective_id or "") or version.artifact_ids.get("version")
                parents = [parent] if parent else []
        content = {"cycle_id": cycle.cycle_id, "action": a.model_dump(mode="json"),
                   "steps": [{"step_id": s.step_id, "kind": s.kind.value, "reuse_lesson": s.reuse_lesson,
                              "session_action": s.session_action} for s in cycle.steps]}
        return self._store(cycle, "learning_cycle", content, parents)

    def _store_outcome_artifact(self, cycle: LearningCycle, outcome: CycleOutcome) -> Artifact:
        """LEARNING_CYCLE (outcome): derives from the cycle artifact, the lesson it used and what the action ended in
        (the session's or the evaluation's next LEARNING_ACTION)."""
        ids = outcome.artifact_ids
        parents = [cycle.artifact_ids["cycle"]] + [ids[k] for k in ("lesson", "learning_action") if k in ids]
        if "learning_action" not in ids and "learner_evaluation" in ids:
            parents.append(ids["learner_evaluation"])
        content = {"cycle_id": cycle.cycle_id, "action_id": cycle.action.action_id,
                   "steps": [{"kind": s.kind.value, "child_id": s.child_id, "reused": s.reused} for s in cycle.steps],
                   "outcome": outcome.model_dump(mode="json")}
        return self._store(cycle, "learning_cycle_outcome", content, list(dict.fromkeys(parents)))

    def _store(self, cycle: LearningCycle, name: str, content: dict, parents: list[str]) -> Artifact:
        a = cycle.action
        return self._artifacts.store(
            task_id=cycle.cycle_id, name=name, type=ArtifactType.LEARNING_CYCLE, media_type="application/json",
            content=json.dumps(content, indent=2, sort_keys=True).encode("utf-8"), provider="teaching-agent",
            parent_ids=parents, metadata={"cycle_id": cycle.cycle_id, "action": a.action.value,
                                          "goal_id": a.goal_id, "objective_id": a.objective_id,
                                          "concept_id": a.concept_id},
            scope=ExecutionScope(events=self._events, usage=UsageLedger(), task_id=cycle.cycle_id, node_id=NODE))

    # --- persistence and the outbox ------------------------------------------------------------------------------

    async def _load(self, cycle_id: str) -> LearningCycle:
        cycle = self.cycle(cycle_id)
        if cycle.pending_events:
            cycle = await self._publish(cycle)  # what a crash left unpublished
        return cycle

    async def _commit(self, cycle: LearningCycle, change: LearningCycle | None) -> LearningCycle:
        """One version-checked write, then its events. A stale version is a conflict: another request is driving
        the cycle."""
        if change is None:
            return cycle
        self._repo.apply(change, expected_version=cycle.version)
        return await self._publish(change)

    async def _publish(self, cycle: LearningCycle) -> LearningCycle:
        if not cycle.pending_events:
            return cycle
        for e in cycle.pending_events:
            self._events.emit(e.type, task_id=cycle.cycle_id, node_id=NODE, event_id=e.event_id, **e.data)
        done = engine.published(cycle, {e.event_id for e in cycle.pending_events}, self._clock())
        try:
            self._repo.apply(done, expected_version=cycle.version)
        except CycleConflict:
            return self.cycle(cycle.cycle_id)  # another request moved on; republishing later stores nothing twice
        return done

    # --- views ---------------------------------------------------------------------------------------------------

    async def _view(self, cycle: LearningCycle, *, created: bool = False, replayed: bool = False,
                    with_prompt: bool = True) -> LearningCycleView:
        return LearningCycleView(
            cycle_id=cycle.cycle_id, learner_id=cycle.learner_id, idempotency_key=cycle.idempotency_key,
            status=cycle.status, action=cycle.action, steps=cycle.steps, waiting=cycle.waiting,
            prompt=await self._prompt(cycle) if with_prompt else None, failure=cycle.failure, outcome=cycle.outcome,
            artifact_ids=cycle.artifact_ids, created=created, replayed=replayed, created_at=cycle.created_at,
            updated_at=cycle.updated_at, completed_at=cycle.completed_at,
            busy_until=cycle.lease.until if cycle.lease is not None and cycle.lease.until > self._clock()
            and cycle.status in ACTIVE_CYCLE_STATUSES else None)

    async def _prompt(self, cycle: LearningCycle) -> LearnerPrompt | None:
        """The learner's view of the waiting point: never an answer key."""
        w = cycle.waiting
        if cycle.status != CycleStatus.WAITING or w is None:
            return None
        if w.kind == "session_answer":
            view = await self._teaching.view(w.child_id)
            turns = []
            for t in reversed(view.turns):
                if t.speaker == Speaker.LEARNER:
                    break
                turns.append(t)
            return LearnerPrompt(kind="SESSION_QUESTION", step_id=w.step_id, question=view.state.waiting_question,
                                 teacher_turns=list(reversed(turns)))
        task = self._tasks.get(w.child_id)
        sheet = task.waiting.prompt if task.waiting is not None else {}
        return LearnerPrompt(kind="DIAGNOSTIC_QUESTIONS" if w.kind == "diagnostic_answers" else "ASSESSMENT_QUESTIONS",
                             step_id=w.step_id, questions=list(sheet.get("questions", [])))

    def _check_sheet(self, cycle: LearningCycle, data: CycleResponse) -> None:
        """A question sheet's answers must name the questions the learner was shown, each at most once."""
        w = cycle.waiting
        if w is None or w.kind == "session_answer" or data.answers is None:
            return
        task = self._tasks.get(w.child_id)
        sheet = task.waiting.prompt if task.waiting is not None else {}
        asked = {q.get("question_id") for q in sheet.get("questions", [])}
        given = [a.question_id for a in data.answers]
        if not given or len(set(given)) != len(given) or not set(given) <= asked:
            raise InvalidCycleRequest("the answers must name the shown questions, each at most once")

    def _failure(self, kind: FailureKind, message: str | None, step: CycleStep | None, *,
                 retryable: bool = False, category: str | None = None) -> CycleFailure:
        return CycleFailure(kind=kind, retryable=retryable, message=(message or kind.value)[:500],
                            step_id=step.step_id if step else None, category=category, at=self._clock())

    def _failed(self, kind: FailureKind, retryable: bool, message: str, step: CycleStep, *,
                category: str | None = None) -> Observation:
        return Observation("failed", failure=self._failure(kind, message, step, retryable=retryable,
                                                           category=category))


def _check_shape(cycle: LearningCycle, data: CycleResponse) -> None:
    w = cycle.waiting
    assert w is not None
    if w.kind == "session_answer":
        if data.answer is None:
            raise InvalidCycleRequest("the learner is asked a session question: send `answer`")
    elif data.answers is None or data.kind != "answer":
        raise InvalidCycleRequest("the learner is asked a question sheet: send `answers`")


def _digest(data: CycleResponse) -> str:
    body = {"kind": data.kind, "answer": data.answer,
            "answers": [a.model_dump(mode="json") for a in data.answers] if data.answers is not None else None}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
