"""Workflow engine: dependency-ordered execution with retries, timeouts, branching, parallel groups,
human waits, checkpoints after every node, and resume from the last checkpoint."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.agents.base import AgentOutputError
from app.agents.registry import AgentRegistry
from app.observability.redaction import redact_text
from app.observability.scope import ExecutionScope
from app.providers.llm.router import AllProvidersFailed, ModelRouter
from app.schemas.common import utcnow
from app.schemas.events import EventType
from app.schemas.task import Task, TaskControl, TaskResult, WaitRequest
from app.schemas.workflow import NodeState, NodeStatus, WorkflowState
from app.tools.base import ToolTransientError
from app.tools.manager import ToolManager
from app.utils.retry import retry_async
from app.runtime.failures import classify
from app.runtime.workflow.nodes import (
    ConditionalNode,
    HumanApprovalNode,
    Node,
    NodeRuntime,
    ParallelNode,
    StateView,
)

RETRYABLE = (TimeoutError, AllProvidersFailed, ToolTransientError, AgentOutputError)
DONE = frozenset({NodeStatus.COMPLETED, NodeStatus.SKIPPED})


class WorkflowDefinitionError(ValueError):
    pass


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    nodes: tuple[Node, ...]
    summarize: Callable[[StateView], TaskResult]
    description: str = ""
    all_nodes: dict[str, Node] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        index: dict[str, Node] = {}
        for node in self.nodes:
            for n in (node, *(node.children if isinstance(node, ParallelNode) else ())):
                if n.id in index:
                    raise WorkflowDefinitionError(f"duplicate node id '{n.id}'")
                index[n.id] = n
        top = {n.id for n in self.nodes}
        for node in self.nodes:
            for dep in (*node.depends_on, *node.after):
                if dep not in index:
                    raise WorkflowDefinitionError(f"node '{node.id}' depends on unknown node '{dep}'")
            if isinstance(node, ConditionalNode):
                for target in (*node.when_true, *node.when_false):
                    if target not in top:
                        raise WorkflowDefinitionError(f"conditional '{node.id}' targets unknown node '{target}'")
        self._check_acyclic()
        object.__setattr__(self, "all_nodes", index)

    def _check_acyclic(self) -> None:
        deps = {n.id: set(n.depends_on) | set(n.after) for n in self.nodes}
        visiting, done = set(), set()

        def visit(nid: str) -> None:
            if nid in done:
                return
            if nid in visiting:
                raise WorkflowDefinitionError(f"cycle through node '{nid}'")
            visiting.add(nid)
            for d in deps.get(nid, ()):
                visit(d)
            visiting.discard(nid)
            done.add(nid)

        for nid in deps:
            visit(nid)

    def initial_state(self) -> WorkflowState:
        return WorkflowState(workflow_id=self.id, node_states={nid: NodeState() for nid in self.all_nodes})


class EngineHooks(Protocol):
    """How the engine talks to whoever owns the task (the orchestrator)."""

    def checkpoint(self, state: WorkflowState, current_step: str | None) -> None: ...

    def control(self) -> TaskControl: ...

    def node_started(self, node: Node) -> None: ...

    def node_finished(self, node: Node, state: NodeState) -> None: ...


@dataclass
class EngineOutcome:
    status: Literal["completed", "waiting", "paused", "cancelled", "failed"]
    wait: WaitRequest | None = None
    error: str | None = None
    failed_node: str | None = None
    category: str | None = None  # FailureCategory value
    stage: str | None = None


class WorkflowEngine:
    def __init__(self, agents: AgentRegistry, tools: ToolManager, router: ModelRouter) -> None:
        self._agents = agents
        self._tools = tools
        self._router = router

    async def run(
        self, definition: WorkflowDefinition, task: Task, scope: ExecutionScope, hooks: EngineHooks
    ) -> EngineOutcome:
        state = task.workflow or definition.initial_state()
        for ns in state.node_states.values():
            if ns.status == NodeStatus.RUNNING:  # interrupted mid-node: run it again from its own progress
                ns.status = NodeStatus.PENDING
        view = StateView(state, task)

        while True:
            control = hooks.control()
            if control.cancel_requested:
                return EngineOutcome(status="cancelled")
            if control.pause_requested:
                return EngineOutcome(status="paused")

            node = self._next_ready(definition, state)
            if node is None:
                pending = [nid for nid, ns in state.node_states.items() if ns.status not in DONE]
                if pending:
                    return EngineOutcome(status="failed", error=f"workflow stalled; unfinished nodes {pending}")
                hooks.checkpoint(state, None)
                return EngineOutcome(status="completed")

            ns = state.node_states[node.id]
            if any(state.node_states[d].status == NodeStatus.SKIPPED for d in node.depends_on):
                ns.status = NodeStatus.SKIPPED
                scope.for_node(node.id).emit(EventType.NODE_SKIPPED, kind=node.kind, reason="dependency skipped")
                hooks.checkpoint(state, node.id)
                continue

            try:
                result = await self._run_node(node, state, view, scope, hooks)
            except Exception as exc:  # BaseException (process kill, cancellation) propagates untouched
                ns.status = NodeStatus.FAILED
                ns.error = redact_text(f"{type(exc).__name__}: {exc}")
                hooks.checkpoint(state, node.id)
                category, stage = classify(exc, node.id)
                return EngineOutcome(status="failed", error=ns.error, failed_node=node.id, category=category.value,
                                     stage=stage.value)
            if result is not None:
                return result

    def _next_ready(self, definition: WorkflowDefinition, state: WorkflowState) -> Node | None:
        for node in definition.nodes:
            ns = state.node_states[node.id]
            if ns.status in DONE:
                continue
            if all(state.node_states[d].status in DONE for d in (*node.depends_on, *node.after)):
                return node
        return None

    async def _run_node(
        self, node: Node, state: WorkflowState, view: StateView, scope: ExecutionScope, hooks: EngineHooks
    ) -> EngineOutcome | None:
        """Run one top-level node. Returns an outcome only when the workflow must stop (waiting)."""
        ns = state.node_states[node.id]
        node_scope = scope.for_node(node.id)

        if isinstance(node, HumanApprovalNode) and ns.status == NodeStatus.WAITING and ns.human_input is None:
            return EngineOutcome(status="waiting", wait=await self._wait_request(node, state, view, node_scope, hooks))

        hooks.node_started(node)
        result = await self._execute(node, state, view, node_scope, hooks)
        if result.wait is not None:
            ns.status = NodeStatus.WAITING
            hooks.checkpoint(state, node.id)
            if isinstance(node, HumanApprovalNode) and node.wait_event:
                node_scope.emit(node.wait_event, kind=node.wait_kind)
            return EngineOutcome(status="waiting", wait=result.wait)
        for target in result.skip:
            target_state = state.node_states[target]
            if target_state.status not in DONE:
                target_state.status = NodeStatus.SKIPPED
                scope.for_node(target).emit(EventType.NODE_SKIPPED, reason=f"branch not taken at {node.id}")
        hooks.checkpoint(state, node.id)
        hooks.node_finished(node, ns)
        return None

    async def _wait_request(self, node, state, view, scope, hooks) -> WaitRequest:
        rt = self._runtime(node, state, view, scope, hooks)
        result = await node.execute(rt)
        assert result.wait is not None
        return result.wait

    async def _execute(self, node: Node, state: WorkflowState, view: StateView, scope: ExecutionScope,
                       hooks: EngineHooks):
        ns = state.node_states[node.id]
        rt = self._runtime(node, state, view, scope, hooks)

        async def attempt(n: int):
            ns.status = NodeStatus.RUNNING
            ns.attempts += 1
            ns.started_at = ns.started_at or utcnow()
            scope.emit(EventType.NODE_STARTED, kind=node.kind, attempt=ns.attempts)
            started = time.perf_counter()
            try:
                if node.timeout_seconds:
                    result = await asyncio.wait_for(node.execute(rt), node.timeout_seconds)
                else:
                    result = await node.execute(rt)
            except Exception as exc:
                ns.error = redact_text(f"{type(exc).__name__}: {exc}")
                scope.emit(EventType.NODE_FAILED, kind=node.kind, attempt=ns.attempts, error=ns.error[:1000])
                raise
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            ns.duration_ms = round((ns.duration_ms or 0) + elapsed, 3)
            if result.wait is not None:
                ns.attempts -= 1  # asking for human input is not an execution attempt
            else:
                ns.status = NodeStatus.COMPLETED
                ns.error = None
                ns.finished_at = utcnow()
                if result.output is not None:
                    ns.output = result.output.model_dump(mode="json")
                    ns.output_type = type(result.output).__name__
                if node.id not in state.execution_order:
                    state.execution_order.append(node.id)
                scope.emit(EventType.NODE_FINISHED, kind=node.kind, duration_ms=elapsed,
                           output_type=ns.output_type)
            return result

        def on_retry(n: int, exc: BaseException) -> None:
            scope.emit(EventType.NODE_RETRY, attempt=n, error=f"{type(exc).__name__}: {exc}"[:1000])

        return await retry_async(attempt, node.retry, retry_on=RETRYABLE, on_retry=on_retry)

    def _runtime(self, node: Node, state: WorkflowState, view: StateView, scope: ExecutionScope,
                 hooks: EngineHooks) -> NodeRuntime:
        async def run_child(child: Node) -> NodeState:
            child_state = state.node_states[child.id]
            if child_state.status == NodeStatus.COMPLETED:
                return child_state
            child_scope = scope.for_node(child.id)
            result = await self._execute(child, state, view, child_scope, hooks)
            if result.wait is not None:
                raise WorkflowDefinitionError("human input nodes cannot run inside a parallel group")
            hooks.checkpoint(state, child.id)
            return child_state

        return NodeRuntime(
            agents=self._agents, tools=self._tools, router=self._router, scope=scope, view=view,
            node_state=state.node_states[node.id], checkpoint=lambda: hooks.checkpoint(state, node.id),
            run_child=run_child, cancel_requested=lambda: hooks.control().cancel_requested,
        )

    async def release(self, definition: WorkflowDefinition, task: Task, scope: ExecutionScope,
                      hooks: EngineHooks) -> list[str]:
        """Let every node the task was waiting in release its outside work (a cancelled task). Returns their ids."""
        state = task.workflow or definition.initial_state()
        view = StateView(state, task)
        released = []
        for node in definition.nodes:
            if state.node_states[node.id].status == NodeStatus.WAITING and not isinstance(node, HumanApprovalNode):
                await node.on_cancel(self._runtime(node, state, view, scope.for_node(node.id), hooks))
                hooks.checkpoint(state, node.id)
                released.append(node.id)
        return released
