"""Orchestrator: owns the task lifecycle (create, plan, execute, wait, pause, resume, cancel, recover).

It contains no educational logic: request interpretation is an agent, the workflow is a template,
and every step runs in the workflow engine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

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
from app.schemas.lesson import InterpretRequest, LessonRequest
from app.schemas.task import Task, TaskControl, TaskError, TaskPlan, TaskStatus
from app.schemas.usage import TaskBudget
from app.schemas.workflow import NodeState, NodeStatus, WorkflowState

NodeObserver = Callable[[Task, str], None]

RESUMABLE = frozenset({TaskStatus.CREATED, TaskStatus.PLANNING, TaskStatus.PAUSED, TaskStatus.RUNNING,
                       TaskStatus.REVIEWING, TaskStatus.FAILED})


class InvalidInput(ValueError):
    pass


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

    # --- commands ----------------------------------------------------------------------------

    def create_task(self, *, request: str, learner_id: str, user_id: str) -> Task:
        task = Task(task_id=new_id("task"), user_id=user_id, learner_id=learner_id, request=request)
        self._tasks.save(task)
        self._events.emit(EventType.TASK_CREATED, task_id=task.task_id, learner_id=learner_id, request=request)
        return task

    def create_planned_task(self, *, request: str, learner_id: str, user_id: str, workflow_id: str,
                            lesson_request: LessonRequest, inputs: dict[str, str],
                            metadata: dict | None = None) -> Task:
        """Create a task whose workflow is already known (no request interpretation needed). `metadata` may carry
        the task's budget (see TaskBudget)."""
        template = self._planner.template(workflow_id)
        definition = template.build(lesson_request)
        estimate = self._planner.estimate(template)
        task = Task(task_id=new_id("task"), user_id=user_id, learner_id=learner_id, request=request,
                    metadata=dict(metadata or {}))
        task.plan = TaskPlan(lesson_request=lesson_request, workflow_id=workflow_id, steps=list(definition.all_nodes),
                             estimated_cost_usd=estimate, inputs=inputs)
        task.cost.estimated_cost_usd = estimate
        task.workflow = definition.initial_state()
        self._tasks.save(task)
        self._events.emit(EventType.TASK_CREATED, task_id=task.task_id, learner_id=learner_id, request=request)
        self._events.emit(EventType.TASK_PLANNED, task_id=task.task_id, workflow_id=workflow_id,
                          estimated_cost_usd=estimate, **inputs)
        return task

    async def run(self, task_id: str) -> Task:
        """Plan (if needed) and execute until the task completes, waits, pauses, or fails."""
        task = self._tasks.get(task_id)
        ledger = UsageLedger(task.cost, budget=TaskBudget.of(task.metadata))
        scope = ExecutionScope(events=self._events, usage=ledger, task_id=task.task_id)
        if task.plan is None:
            if not await self._plan(task, scope, ledger):
                return task
        elif task.status in (TaskStatus.CREATED, TaskStatus.PAUSED) and not task.metadata.get("started"):
            transition(task, TaskStatus.PLANNING)  # pre-planned task: planning already happened at creation
        return await self._execute(task, scope, ledger)

    async def submit_input(self, task_id: str, node_id: str, payload: dict) -> Task:
        task = self._tasks.get(task_id)
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
        if node.submitted_event:
            self._events.emit(node.submitted_event, task_id=task_id, node_id=node_id)
        self._events.emit(EventType.TASK_RESUMED, task_id=task_id, node_id=node_id, reason="input received")
        self._save(task)
        return await self.run(task_id)

    async def resume(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task.status not in RESUMABLE:
            raise InvalidTransition(f"task {task_id} in status {task.status.value} cannot be resumed")
        task.control = TaskControl()
        if task.workflow is not None:
            for ns in task.workflow.node_states.values():
                if ns.status == NodeStatus.FAILED:
                    ns.status = NodeStatus.PENDING
        self._events.emit(EventType.TASK_RESUMED, task_id=task_id, from_status=task.status.value)
        self._tasks.save(task)
        return await self.run(task_id)

    def invalidate(self, task_id: str, node_ids: Sequence[str]) -> list[str]:
        """Mark nodes (and everything downstream of them) to run again on the next resume: for a task whose stored
        artifacts turned out to be missing or corrupt. Returns the node ids that were reset."""
        task = self._tasks.get(task_id)
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
        self._tasks.save(task)
        ordered = [nid for nid in definition.all_nodes if nid in reset]
        self._events.emit(EventType.TASK_RESUMED, task_id=task_id, reason="artifacts invalid", reset_nodes=ordered)
        return ordered

    def request_pause(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            raise InvalidTransition(f"task {task_id} is already {task.status.value}")
        if task.status == TaskStatus.CREATED:
            transition(task, TaskStatus.PAUSED)
            self._events.emit(EventType.TASK_PAUSED, task_id=task_id, at_step=None)
        # a running task stops at the next node boundary; a waiting task stops once its input arrives
        task.control.pause_requested = True
        self._tasks.save(task)
        return task

    def request_cancel(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            raise InvalidTransition(f"task {task_id} is already {task.status.value}")
        if task.status in (TaskStatus.CREATED, TaskStatus.WAITING, TaskStatus.PAUSED, TaskStatus.FAILED):
            task.status = TaskStatus.CANCELLED  # nothing is executing: cancel immediately
            task.waiting = None
            self._events.emit(EventType.TASK_CANCELLED, task_id=task_id)
        task.control.cancel_requested = True
        self._tasks.save(task)
        return task

    # --- internals ---------------------------------------------------------------------------

    async def _plan(self, task: Task, scope: ExecutionScope, ledger: UsageLedger) -> bool:
        transition(task, TaskStatus.PLANNING)
        self._save(task, ledger)
        try:
            interpreter = self._agents.get(self._interpreter)
            ctx = AgentContext(router=self._router, tools=self._tools, scope=scope.for_node("interpret_request"))
            lesson_request = await interpreter.execute(
                InterpretRequest(request=task.request, learner_id=task.learner_id), ctx
            )
            assert isinstance(lesson_request, LessonRequest)
            template = self._planner.select(lesson_request)
            definition = template.build(lesson_request)
        except Exception as exc:
            self._fail(task, ledger, "planning", exc, node_id="interpret_request")
            return False
        estimate = self._planner.estimate(template)
        task.plan = TaskPlan(lesson_request=lesson_request, workflow_id=template.id,
                             steps=list(definition.all_nodes), estimated_cost_usd=estimate)
        ledger.summary.estimated_cost_usd = estimate
        task.workflow = definition.initial_state()
        self._events.emit(EventType.TASK_PLANNED, task_id=task.task_id, workflow_id=template.id,
                          estimated_cost_usd=estimate, subject=lesson_request.subject, topic=lesson_request.topic)
        self._save(task, ledger)
        return True

    async def _execute(self, task: Task, scope: ExecutionScope, ledger: UsageLedger) -> Task:
        assert task.plan is not None
        definition = self._definition(task)
        first_start = not task.metadata.get("started")
        transition(task, TaskStatus.RUNNING)
        task.metadata["started"] = True
        self._save(task, ledger)
        if first_start:
            self._events.emit(EventType.TASK_STARTED, task_id=task.task_id, workflow_id=definition.id)

        hooks = _Hooks(self, task, ledger)
        outcome = await self._engine.run(definition, task, scope, hooks)
        self._finish(task, ledger, definition, outcome)
        return task

    def _finish(self, task: Task, ledger: UsageLedger, definition: WorkflowDefinition, outcome: EngineOutcome) -> None:
        if outcome.status == "completed":
            assert task.workflow is not None
            try:
                task.result = definition.summarize(StateView(task.workflow, task))
            except Exception as exc:
                self._fail(task, ledger, "summarize", exc)
                return
            task.artifact_ids = [a.artifact_id for a in task.result.artifacts]
            transition(task, TaskStatus.COMPLETED)
            task.current_step = None
            self._save(task, ledger)
            self._events.emit(EventType.TASK_COMPLETED, task_id=task.task_id,
                              actual_cost_usd=task.cost.actual_cost_usd,
                              estimated_cost_usd=task.cost.estimated_cost_usd,
                              total_tokens=task.cost.token_usage.total_tokens, artifacts=len(task.artifact_ids))
        elif outcome.status == "waiting":
            task.waiting = outcome.wait
            transition(task, TaskStatus.WAITING)
            self._save(task, ledger)
            assert outcome.wait is not None
            self._events.emit(EventType.TASK_WAITING, task_id=task.task_id, node_id=outcome.wait.node_id,
                              kind=outcome.wait.kind)
        elif outcome.status == "paused":
            task.control.pause_requested = False
            transition(task, TaskStatus.PAUSED)
            self._save(task, ledger, keep_control=True)
            self._events.emit(EventType.TASK_PAUSED, task_id=task.task_id, at_step=task.current_step)
        elif outcome.status == "cancelled":
            transition(task, TaskStatus.CANCELLED)
            self._save(task, ledger)
            self._events.emit(EventType.TASK_CANCELLED, task_id=task.task_id)
        else:
            self._fail(task, ledger, "workflow", RuntimeError(outcome.error or "workflow failed"),
                       node_id=outcome.failed_node, category=outcome.category, stage=outcome.stage)

    def _fail(self, task: Task, ledger: UsageLedger, kind: str, exc: Exception, node_id: str | None = None,
              category: str | None = None, stage: str | None = None) -> None:
        if category is None:
            classified, staged = classify(exc, node_id)
            category, stage = classified.value, staged.value
        task.errors.append(TaskError(kind=kind, message=redact_text(f"{type(exc).__name__}: {exc}"), node_id=node_id,
                                     category=category, stage=stage))
        transition(task, TaskStatus.FAILED)
        self._save(task, ledger)
        self._events.emit(EventType.TASK_FAILED, task_id=task.task_id, node_id=node_id, category=category,
                          error=redact_text(str(exc))[:1000])

    def _definition(self, task: Task) -> WorkflowDefinition:
        assert task.plan is not None
        return self._planner.build(task.plan.workflow_id, task.plan.lesson_request)

    def _save(self, task: Task, ledger: UsageLedger | None = None, keep_control: bool = False) -> None:
        if ledger is not None:
            task.cost = ledger.summary.model_copy(deep=True)
        if not keep_control:
            # pause/cancel requests arrive from other callers; never overwrite them with a stale copy
            try:
                task.control = self._tasks.get(task.task_id).control
            except KeyError:
                pass
        task.updated_at = utcnow()
        self._tasks.save(task)

    def _node_started(self, task: Task, ledger: UsageLedger, node: Node) -> None:
        target = TaskStatus.REVIEWING if isinstance(node, ReviewNode) else TaskStatus.RUNNING
        if task.status != target:
            transition(task, target)
            self._save(task, ledger)

    def _node_finished(self, task: Task, node_id: str) -> None:
        for observer in self._observers:
            observer(task, node_id)


class _Hooks:
    def __init__(self, orchestrator: Orchestrator, task: Task, ledger: UsageLedger) -> None:
        self._o = orchestrator
        self._task = task
        self._ledger = ledger

    def checkpoint(self, state: WorkflowState, current_step: str | None) -> None:
        self._task.workflow = state
        self._task.current_step = current_step
        self._o._save(self._task, self._ledger)

    def control(self) -> TaskControl:
        return self._o._tasks.get(self._task.task_id).control

    def node_started(self, node: Node) -> None:
        self._o._node_started(self._task, self._ledger, node)

    def node_finished(self, node: Node, state: NodeState) -> None:
        self._o._node_finished(self._task, node.id)
