"""Orchestrator: owns the task lifecycle (create, plan, execute, wait, pause, resume, cancel, recover).

It contains no educational logic: request interpretation is an agent, the workflow is a template,
and every step runs in the workflow engine.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta

from app.agents.base import AgentContext
from app.agents.registry import AgentRegistry
from app.observability.events import EventBus
from app.observability.redaction import redact_text
from app.observability.scope import ExecutionScope, UsageLedger
from app.providers.llm.router import ModelRouter
from app.runtime.failures import classify
from app.runtime.orchestrator.planner import WorkflowPlanner
from app.runtime.tasks.repository import TaskRepository
from app.runtime.tasks.state_machine import InvalidTransition, transition
from app.runtime.workflow.engine import EngineOutcome, WorkflowDefinition, WorkflowEngine
from app.runtime.workflow.nodes import HumanApprovalNode, Node, ReviewNode, StateView
from app.schemas.common import new_id, utcnow
from app.schemas.events import EventType
from app.schemas.lease import Lease, LeaseLost
from app.schemas.lesson import InterpretRequest, LessonRequest
from app.schemas.task import Task, TaskControl, TaskError, TaskPlan, TaskStatus
from app.schemas.usage import TaskBudget
from app.schemas.workflow import NodeState, NodeStatus, WorkflowState

NodeObserver = Callable[[Task, str], None]
logger = logging.getLogger(__name__)

# How long a worker's claim on a task lasts without renewal. A live worker renews it every third of this (heartbeat),
# so it only runs out when the worker died or hung; then another request may claim the task and resume it.
TASK_LEASE = timedelta(minutes=5)

RESUMABLE = frozenset({TaskStatus.CREATED, TaskStatus.PLANNING, TaskStatus.PAUSED, TaskStatus.RUNNING,
                       TaskStatus.REVIEWING, TaskStatus.FAILED})


class InvalidInput(ValueError):
    pass


class TaskBusy(InvalidTransition):
    """Another worker holds the task's live lease: this request executes nothing. Fixed text."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"task {task_id} is being executed by another request: read it again later")
        self.task_id = task_id


class Orchestrator:
    def __init__(
        self,
        *,
        tasks: TaskRepository,
        engine: WorkflowEngine,
        planner: WorkflowPlanner,
        agents: AgentRegistry,
        tools,
        router: ModelRouter,
        events: EventBus,
        interpreter_agent: str = "request_interpreter",
        observers: Sequence[NodeObserver] = (),
        lease_ttl: timedelta = TASK_LEASE,
        heartbeat_seconds: float | None = None,
    ) -> None:
        self._tasks = tasks
        self._engine = engine
        self._planner = planner
        self._agents = agents
        self._tools = tools
        self._router = router
        self._events = events
        self._interpreter = interpreter_agent
        self._observers = list(observers)
        self._lease_ttl = lease_ttl
        self._heartbeat = heartbeat_seconds if heartbeat_seconds is not None else lease_ttl.total_seconds() / 3

    # --- commands ----------------------------------------------------------------------------

    def create_task(self, *, request: str, learner_id: str, user_id: str) -> Task:
        task = Task(task_id=new_id("task"), user_id=user_id, learner_id=learner_id, request=request)
        self._tasks.create(task)
        self._events.emit(EventType.TASK_CREATED, task_id=task.task_id, learner_id=learner_id, request=request)
        return task

    def create_planned_task(self, *, request: str, learner_id: str, user_id: str, workflow_id: str,
                            lesson_request: LessonRequest, inputs: dict[str, str],
                            metadata: dict | None = None, task_id: str | None = None) -> Task:
        """Create a task whose workflow is already known (no request interpretation needed). `metadata` may carry
        the task's budget (see TaskBudget). A deterministic `task_id` makes creation idempotent: when that task
        exists it is returned unchanged, so concurrent callers share one task instead of each creating one."""
        template = self._planner.template(workflow_id)
        definition = template.build(lesson_request)
        estimate = self._planner.estimate(template)
        task = Task(task_id=task_id or new_id("task"), user_id=user_id, learner_id=learner_id, request=request,
                    metadata=dict(metadata or {}))
        task.plan = TaskPlan(lesson_request=lesson_request, workflow_id=workflow_id, steps=list(definition.all_nodes),
                             estimated_cost_usd=estimate, inputs=inputs)
        task.cost.estimated_cost_usd = estimate
        task.workflow = definition.initial_state()
        if not self._tasks.create(task):
            return self._tasks.get(task.task_id)
        self._events.emit(EventType.TASK_CREATED, task_id=task.task_id, learner_id=learner_id, request=request)
        self._events.emit(EventType.TASK_PLANNED, task_id=task.task_id, workflow_id=workflow_id,
                          estimated_cost_usd=estimate, **inputs)
        return task

    async def run(self, task_id: str) -> Task:
        """Plan (if needed) and execute until the task completes, waits, pauses, or fails. Raises TaskBusy (and
        executes nothing) while another worker holds the task's lease."""
        async with self._held(task_id) as lease:
            return await self._run(self._tasks.get(task_id), lease)

    async def _run(self, task: Task, lease: Lease) -> Task:
        ledger = UsageLedger(task.cost, budget=TaskBudget.of(task.metadata))
        scope = self._scope(task, ledger, lease)
        if task.plan is None:
            if not await self._plan(task, scope, ledger, lease):
                return task
        elif task.status in (TaskStatus.CREATED, TaskStatus.PAUSED) and not task.metadata.get("started"):
            transition(task, TaskStatus.PLANNING)  # pre-planned task: planning already happened at creation
        return await self._execute(task, scope, ledger, lease)

    async def submit_input(self, task_id: str, node_id: str, payload: dict) -> Task:
        async with self._held(task_id) as lease:  # checked under the lease: a concurrent submit cannot also pass
            return await self._submit_input(self._tasks.get(task_id), node_id, payload, lease)

    async def _submit_input(self, task: Task, node_id: str, payload: dict, lease: Lease) -> Task:
        task_id = task.task_id
        if task.status != TaskStatus.WAITING or task.waiting is None or task.waiting.node_id != node_id:
            raise InvalidTransition(f"task {task_id} is not waiting for input at '{node_id}'")
        definition = self._definition(task)
        node = definition.all_nodes[node_id]
        assert isinstance(node, HumanApprovalNode)
        assert task.workflow is not None
        try:
            data = node.response_model.model_validate(payload)
            if node.validate_input is not None:
                node.validate_input(StateView(task.workflow, task), data)
        except ValueError as exc:  # includes pydantic ValidationError
            raise InvalidInput(str(exc)) from exc
        task.workflow.node_states[node_id].human_input = data.model_dump(mode="json")
        task.waiting = None
        self._save(task, None, lease)
        if node.submitted_event:
            self._events.emit(node.submitted_event, task_id=task_id, node_id=node_id, fence=lease)
        self._events.emit(EventType.TASK_RESUMED, task_id=task_id, node_id=node_id, reason="input received",
                          fence=lease)
        return await self._run(task, lease)

    def waiting_on_work(self, task: Task) -> bool:
        """Whether a WAITING task waits on outside work (e.g. a video generation job), not on a person. Such a task
        is resumed (it checks the work again) rather than given input."""
        if task.status != TaskStatus.WAITING or task.waiting is None or task.plan is None:
            return False
        node = self._definition(task).all_nodes.get(task.waiting.node_id)
        return node is not None and not isinstance(node, HumanApprovalNode)

    async def resume(self, task_id: str) -> Task:
        """Continue a task from its last checkpoint. A RUNNING or REVIEWING task is resumable only once its worker's
        lease has run out (the worker died or hung): while it is live this raises TaskBusy and runs nothing."""
        async with self._held(task_id) as lease:
            task = self._tasks.get(task_id)
            if task.status not in RESUMABLE and not self.waiting_on_work(task):
                raise InvalidTransition(f"task {task_id} in status {task.status.value} cannot be resumed")
            task.control = TaskControl()
            task.waiting = None
            if task.workflow is not None:
                for ns in task.workflow.node_states.values():
                    if ns.status == NodeStatus.FAILED:
                        ns.status = NodeStatus.PENDING
            from_status = task.status.value
            self._save(task, None, lease, clear_control=("pause", "cancel"))
            self._events.emit(EventType.TASK_RESUMED, task_id=task_id, from_status=from_status, fence=lease)
            return await self._run(task, lease)

    def invalidate(self, task_id: str, node_ids: Sequence[str]) -> list[str]:
        """Mark nodes (and everything downstream of them) to run again on the next resume: for a task whose stored
        artifacts turned out to be missing or corrupt. Returns the node ids that were reset."""
        with self._held_now(task_id) as lease:
            return self._invalidate(self._tasks.get(task_id), node_ids, lease)

    def _invalidate(self, task: Task, node_ids: Sequence[str], lease: Lease) -> list[str]:
        task_id = task.task_id
        if task.status not in RESUMABLE or task.status == TaskStatus.CREATED:
            raise InvalidTransition(f"task {task_id} in status {task.status.value} cannot be repaired")
        assert task.workflow is not None
        definition = self._definition(task)
        dependents: dict[str, set[str]] = {nid: set() for nid in definition.all_nodes}
        for node in definition.all_nodes.values():
            for upstream in (*node.depends_on, *node.after):
                dependents[upstream].add(node.id)
        reset: set[str] = set()
        frontier = [nid for nid in node_ids if nid in definition.all_nodes]
        while frontier:
            nid = frontier.pop()
            if nid not in reset:
                reset.add(nid)
                frontier.extend(dependents[nid])
        for nid in reset:
            task.workflow.node_states[nid] = NodeState()
        task.workflow.execution_order = [n for n in task.workflow.execution_order if n not in reset]
        self._save(task, None, lease)
        ordered = [nid for nid in definition.all_nodes if nid in reset]
        self._events.emit(EventType.TASK_RESUMED, task_id=task_id, reason="artifacts invalid", reset_nodes=ordered,
                          fence=lease)
        return ordered

    def request_pause(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            raise InvalidTransition(f"task {task_id} is already {task.status.value}")
        # The request is recorded first, in its own column: a worker that starts meanwhile still sees it.
        # A running task stops at the next node boundary; a waiting task stops once its input arrives.
        self._tasks.request_control(task_id, pause=True)
        if task.status == TaskStatus.CREATED:
            with self._try_hold(task_id) as lease:  # nobody executes it: pause it now (else the worker will)
                if lease is not None:
                    task = self._tasks.get(task_id)
                    if task.status == TaskStatus.CREATED:
                        transition(task, TaskStatus.PAUSED)
                        self._save(task, None, lease)
                        self._events.emit(EventType.TASK_PAUSED, task_id=task_id, at_step=None, fence=lease)
        return self._tasks.get(task_id)

    def request_cancel(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            raise InvalidTransition(f"task {task_id} is already {task.status.value}")
        self._tasks.request_control(task_id, cancel=True)  # an executing worker stops at its next node boundary
        idle = (TaskStatus.CREATED, TaskStatus.WAITING, TaskStatus.PAUSED, TaskStatus.FAILED)
        if task.status in idle:
            with self._try_hold(task_id) as lease:  # nothing is executing: cancel immediately
                if lease is not None:
                    task = self._tasks.get(task_id)
                    if task.status in idle:
                        task.status = TaskStatus.CANCELLED
                        task.waiting = None
                        self._save(task, None, lease)
                        self._events.emit(EventType.TASK_CANCELLED, task_id=task_id, fence=lease)
        return self._tasks.get(task_id)

    async def release(self, task_id: str) -> list[str]:
        """For a cancelled task: let the nodes it was waiting in release their outside work (cancel provider jobs).
        Returns the node ids released. Artifacts already stored are untouched."""
        async with self._held(task_id) as lease:
            task = self._tasks.get(task_id)
            if task.status != TaskStatus.CANCELLED or task.plan is None or task.workflow is None:
                return []
            ledger = UsageLedger(task.cost, budget=TaskBudget.of(task.metadata))
            scope = self._scope(task, ledger, lease)
            released = await self._engine.release(self._definition(task), task, scope,
                                                  _Hooks(self, task, ledger, lease))
            self._save(task, ledger, lease)
            return released

    # --- the execution lease -----------------------------------------------------------------------------------

    @asynccontextmanager
    async def _held(self, task_id: str) -> AsyncIterator[Lease]:
        """Claim the task's lease for this request (TaskBusy while another worker's is live), renew it while the
        work runs, and give it back however the work ends. Every write made under it is fenced by the storage
        layer: once the lease is lost, nothing more of this worker's is committed."""
        lease = self._claim(task_id)
        beat = asyncio.create_task(self._renew_until_lost(lease)) if self._heartbeat > 0 else None
        try:
            yield lease
        except LeaseLost:
            self._lease_lost(lease)
            raise
        finally:
            if beat is not None:
                beat.cancel()
            self._tasks.release(lease)

    @contextmanager
    def _held_now(self, task_id: str) -> Iterator[Lease]:
        """`_held` for a short synchronous change (no heartbeat)."""
        lease = self._claim(task_id)
        try:
            yield lease
        except LeaseLost:
            self._lease_lost(lease)
            raise
        finally:
            self._tasks.release(lease)

    @contextmanager
    def _try_hold(self, task_id: str) -> Iterator[Lease | None]:
        lease = self._tasks.claim(task_id, new_id("worker"), self._lease_ttl)
        try:
            yield lease
        finally:
            if lease is not None:
                self._tasks.release(lease)

    def _scope(self, task: Task, ledger: UsageLedger, lease: Lease) -> ExecutionScope:
        def check() -> None:
            if not self._tasks.holds(lease):
                raise LeaseLost(lease)

        return ExecutionScope(events=self._events, usage=ledger, task_id=task.task_id, fence=lease, fence_check=check)

    def _claim(self, task_id: str) -> Lease:
        lease = self._tasks.claim(task_id, new_id("worker"), self._lease_ttl)
        if lease is None:
            raise TaskBusy(task_id)
        return lease

    async def _renew_until_lost(self, lease: Lease) -> None:
        while True:
            await asyncio.sleep(self._heartbeat)
            renewed = self._tasks.renew(lease, self._lease_ttl)
            if renewed is None:  # expired or taken over: this worker's next write is refused
                return
            lease = renewed

    def _lease_lost(self, lease: Lease) -> None:
        """Internal diagnostics for a stale worker whose write was refused (stored unfenced: it is about the loss)."""
        logger.warning("task %s: worker %s lost its lease (fence %s); its writes were refused",
                       lease.resource_id, lease.owner, lease.fence)
        self._events.emit(EventType.TASK_LEASE_LOST, task_id=lease.resource_id, owner=lease.owner,
                          fence=None, stale_fence=lease.fence)

    # --- internals ---------------------------------------------------------------------------

    async def _plan(self, task: Task, scope: ExecutionScope, ledger: UsageLedger, lease: Lease) -> bool:
        transition(task, TaskStatus.PLANNING)
        self._save(task, ledger, lease)
        try:
            interpreter = self._agents.get(self._interpreter)
            ctx = AgentContext(router=self._router, tools=self._tools, scope=scope.for_node("interpret_request"))
            lesson_request = await interpreter.execute(
                InterpretRequest(request=task.request, learner_id=task.learner_id), ctx
            )
            assert isinstance(lesson_request, LessonRequest)
            template = self._planner.select(lesson_request)
            definition = template.build(lesson_request)
        except LeaseLost:
            raise
        except Exception as exc:
            self._fail(task, ledger, lease, "planning", exc, node_id="interpret_request")
            return False
        estimate = self._planner.estimate(template)
        task.plan = TaskPlan(lesson_request=lesson_request, workflow_id=template.id,
                             steps=list(definition.all_nodes), estimated_cost_usd=estimate)
        ledger.summary.estimated_cost_usd = estimate
        task.workflow = definition.initial_state()
        self._events.emit(EventType.TASK_PLANNED, task_id=task.task_id, workflow_id=template.id,
                          estimated_cost_usd=estimate, subject=lesson_request.subject, topic=lesson_request.topic,
                          fence=lease)
        self._save(task, ledger, lease)
        return True

    async def _execute(self, task: Task, scope: ExecutionScope, ledger: UsageLedger, lease: Lease) -> Task:
        assert task.plan is not None
        definition = self._definition(task)
        first_start = not task.metadata.get("started")
        transition(task, TaskStatus.RUNNING)
        task.metadata["started"] = True
        self._save(task, ledger, lease)
        if first_start:
            self._events.emit(EventType.TASK_STARTED, task_id=task.task_id, workflow_id=definition.id, fence=lease)

        hooks = _Hooks(self, task, ledger, lease)
        outcome = await self._engine.run(definition, task, scope, hooks)
        self._finish(task, ledger, lease, definition, outcome)
        return task

    def _finish(self, task: Task, ledger: UsageLedger, lease: Lease, definition: WorkflowDefinition,
                outcome: EngineOutcome) -> None:
        """Record the outcome: the state write comes first and is fenced, so a worker that lost its lease publishes
        no result and no success event."""
        if outcome.status == "completed":
            assert task.workflow is not None
            try:
                task.result = definition.summarize(StateView(task.workflow, task))
            except Exception as exc:
                self._fail(task, ledger, lease, "summarize", exc)
                return
            task.artifact_ids = [a.artifact_id for a in task.result.artifacts]
            transition(task, TaskStatus.COMPLETED)
            task.current_step = None
            self._save(task, ledger, lease)
            self._events.emit(EventType.TASK_COMPLETED, task_id=task.task_id,
                              actual_cost_usd=task.cost.actual_cost_usd,
                              estimated_cost_usd=task.cost.estimated_cost_usd,
                              total_tokens=task.cost.token_usage.total_tokens, artifacts=len(task.artifact_ids),
                              fence=lease)
        elif outcome.status == "waiting":
            task.waiting = outcome.wait
            transition(task, TaskStatus.WAITING)
            self._save(task, ledger, lease)
            assert outcome.wait is not None
            self._events.emit(EventType.TASK_WAITING, task_id=task.task_id, node_id=outcome.wait.node_id,
                              kind=outcome.wait.kind, fence=lease)
        elif outcome.status == "paused":
            task.control.pause_requested = False
            transition(task, TaskStatus.PAUSED)
            self._save(task, ledger, lease, clear_control=("pause",))
            self._events.emit(EventType.TASK_PAUSED, task_id=task.task_id, at_step=task.current_step, fence=lease)
        elif outcome.status == "cancelled":
            transition(task, TaskStatus.CANCELLED)
            self._save(task, ledger, lease)
            self._events.emit(EventType.TASK_CANCELLED, task_id=task.task_id, fence=lease)
        else:
            self._fail(task, ledger, lease, "workflow", RuntimeError(outcome.error or "workflow failed"),
                       node_id=outcome.failed_node, category=outcome.category, stage=outcome.stage)

    def _fail(self, task: Task, ledger: UsageLedger, lease: Lease, kind: str, exc: Exception,
              node_id: str | None = None, category: str | None = None, stage: str | None = None) -> None:
        if category is None:
            classified, staged = classify(exc, node_id)
            category, stage = classified.value, staged.value
        task.errors.append(TaskError(kind=kind, message=redact_text(f"{type(exc).__name__}: {exc}"), node_id=node_id,
                                     category=category, stage=stage))
        transition(task, TaskStatus.FAILED)
        self._save(task, ledger, lease)
        self._events.emit(EventType.TASK_FAILED, task_id=task.task_id, node_id=node_id, category=category,
                          error=redact_text(str(exc))[:1000], fence=lease)

    def _definition(self, task: Task) -> WorkflowDefinition:
        assert task.plan is not None
        return self._planner.build(task.plan.workflow_id, task.plan.lesson_request)

    def _save(self, task: Task, ledger: UsageLedger | None, lease: Lease,
              clear_control: Iterable[str] = ()) -> None:
        """A fenced write of the task (LeaseLost once `lease` is not the current, unexpired one). Pause and cancel
        requests live in their own columns, so this never overwrites one another caller just made."""
        if ledger is not None:
            task.cost = ledger.summary.model_copy(deep=True)
        task.updated_at = utcnow()
        self._tasks.save(task, lease=lease, clear_control=clear_control)

    def _node_started(self, task: Task, ledger: UsageLedger, lease: Lease, node: Node) -> None:
        target = TaskStatus.REVIEWING if isinstance(node, ReviewNode) else TaskStatus.RUNNING
        if task.status != target:
            transition(task, target)
            self._save(task, ledger, lease)

    def _node_finished(self, task: Task, node_id: str) -> None:
        for observer in self._observers:
            observer(task, node_id)


class _Hooks:
    def __init__(self, orchestrator: Orchestrator, task: Task, ledger: UsageLedger, lease: Lease) -> None:
        self._o = orchestrator
        self._task = task
        self._ledger = ledger
        self._lease = lease

    def checkpoint(self, state: WorkflowState, current_step: str | None) -> None:
        self._task.workflow = state
        self._task.current_step = current_step
        self._o._save(self._task, self._ledger, self._lease)

    def control(self) -> TaskControl:
        return self._o._tasks.control(self._task.task_id)

    def node_started(self, node: Node) -> None:
        self._o._node_started(self._task, self._ledger, self._lease, node)

    def node_finished(self, node: Node, state: NodeState) -> None:
        self._o._node_finished(self._task, node.id)
