"""Workflow engine on toy workflows: ordering, branches, skips, parallel groups, retries, timeouts, waits, resume."""

from __future__ import annotations

import asyncio

import pytest

from app.agents.registry import AgentRegistry
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.router import ModelRouter
from app.runtime.workflow.engine import WorkflowDefinition, WorkflowDefinitionError, WorkflowEngine
from app.runtime.workflow.nodes import (
    ConditionalNode,
    HumanApprovalNode,
    Node,
    NodeResult,
    ParallelNode,
    ToolNode,
    TransformNode,
)
from app.schemas.common import RetryPolicy, Schema
from app.schemas.task import Task, TaskControl, TaskResult
from app.schemas.workflow import NodeStatus
from app.tools.base import ToolTransientError
from app.tools.manager import ToolManager
from app.tools.registry import ToolRegistry
from tests.unit.helpers import routing, scope
from tests.unit.test_tools import Echo


class Num(Schema):
    value: int


class Approval(Schema):
    approved: bool


def const(v: int):
    return lambda view: Num(value=v)


def plus(node: str, n: int):
    return lambda view: Num(value=view.output(node, Num).value + n)


def summary(view) -> TaskResult:
    return TaskResult(title="t", artifacts=[], mastery_changes=[], review_verdict="n/a", revisions=0,
                      estimated_level=None)


class Hooks:
    def __init__(self) -> None:
        self.checkpoints = 0
        self.control_state = TaskControl()
        self.started: list[str] = []

    def checkpoint(self, state, current_step) -> None:
        self.checkpoints += 1

    def control(self) -> TaskControl:
        return self.control_state

    def node_started(self, node) -> None:
        self.started.append(node.id)

    def node_finished(self, node, state) -> None:
        pass


def engine() -> WorkflowEngine:
    registry = ToolRegistry()
    registry.register(Echo())
    return WorkflowEngine(AgentRegistry(), ToolManager(registry),
                          ModelRouter(routing(("mock", "m1")), {"mock": MockLLMProvider({})}))


def new_task() -> Task:
    return Task(task_id="t1", user_id="u", learner_id="l", request="r")


async def run(definition: WorkflowDefinition, task: Task | None = None, hooks: Hooks | None = None):
    task = task or new_task()
    task.workflow = task.workflow or definition.initial_state()
    sc, events = scope()
    outcome = await engine().run(definition, task, sc, hooks or Hooks())
    return outcome, task, events


async def test_sequential_dependencies_and_checkpoints() -> None:
    wf = WorkflowDefinition(id="w", summarize=summary, nodes=(
        TransformNode(id="c", fn=plus("b", 10), depends_on=("b",)),
        TransformNode(id="a", fn=const(1)),
        TransformNode(id="b", fn=plus("a", 1), depends_on=("a",)),
    ))
    hooks = Hooks()
    outcome, task, events = await run(wf, hooks=hooks)
    assert outcome.status == "completed"
    assert task.workflow.execution_order == ["a", "b", "c"]
    assert task.workflow.node_states["c"].output == {"value": 12}
    assert hooks.checkpoints >= 3
    assert [e.type for e in events].count("node.finished") == 3


async def test_conditional_skip_cascades_through_hard_dependencies_only() -> None:
    wf = WorkflowDefinition(id="w", summarize=summary, nodes=(
        TransformNode(id="a", fn=const(5)),
        ConditionalNode(id="gate", depends_on=("a",), predicate=lambda v: v.output("a", Num).value > 10,
                        when_true=("big",), when_false=("small",)),
        TransformNode(id="big", depends_on=("gate",), fn=const(100)),
        TransformNode(id="after_big", depends_on=("big",), fn=const(101)),
        TransformNode(id="small", depends_on=("gate",), fn=const(1)),
        TransformNode(id="join", depends_on=("a",), after=("after_big", "small"),
                      fn=lambda v: v.maybe("after_big", Num) or v.output("small", Num)),
    ))
    outcome, task, _ = await run(wf)
    states = task.workflow.node_states
    assert outcome.status == "completed"
    assert states["big"].status == NodeStatus.SKIPPED and states["after_big"].status == NodeStatus.SKIPPED
    assert states["join"].output == {"value": 1}


async def test_parallel_group_runs_children_concurrently() -> None:
    order: list[str] = []

    class Sleepy(Node):
        kind = "test"

        async def execute(self, rt):
            order.append(f"start:{self.id}")
            await asyncio.sleep(0.01)
            order.append(f"end:{self.id}")
            return NodeResult(output=Num(value=1))

    wf = WorkflowDefinition(id="w", summarize=summary, nodes=(
        ParallelNode(id="p", children=(Sleepy(id="x"), Sleepy(id="y"))),
        TransformNode(id="after", depends_on=("p",), fn=lambda v: Num(value=v.output("x", Num).value + v.output("y", Num).value)),
    ))
    outcome, task, _ = await run(wf)
    assert outcome.status == "completed"
    assert order[:2] == ["start:x", "start:y"]
    assert task.workflow.node_states["after"].output == {"value": 2}


async def test_retries_transient_failures_and_times_out() -> None:
    calls = {"n": 0}

    class Flaky(Node):
        kind = "test"

        async def execute(self, rt):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ToolTransientError("blip")
            return NodeResult(output=Num(value=calls["n"]))

    class Slow(Node):
        kind = "test"

        async def execute(self, rt):
            await asyncio.sleep(1)
            return NodeResult(output=Num(value=0))

    ok, task, events = await run(WorkflowDefinition(id="w", summarize=summary, nodes=(
        Flaky(id="f", retry=RetryPolicy(max_attempts=3)),)))
    assert ok.status == "completed" and task.workflow.node_states["f"].attempts == 3
    assert [e.type for e in events].count("node.retry") == 2

    slow, task, _ = await run(WorkflowDefinition(id="w", summarize=summary, nodes=(
        Slow(id="s", timeout_seconds=0.02, retry=RetryPolicy(max_attempts=2)),)))
    assert slow.status == "failed" and slow.failed_node == "s"
    assert task.workflow.node_states["s"].attempts == 2 and "TimeoutError" in task.workflow.node_states["s"].error


async def test_non_retryable_failure_stops_the_workflow() -> None:
    def boom(view):
        raise RuntimeError("bad data")

    outcome, task, _ = await run(WorkflowDefinition(id="w", summarize=summary, nodes=(
        TransformNode(id="x", fn=boom, retry=RetryPolicy(max_attempts=3)),
        TransformNode(id="y", depends_on=("x",), fn=const(1)),
    )))
    assert outcome.status == "failed" and task.workflow.node_states["x"].attempts == 1
    assert task.workflow.node_states["y"].status == NodeStatus.PENDING


async def test_human_wait_and_resume_skip_completed_nodes() -> None:
    counter = {"a": 0}

    def count_a(view):
        counter["a"] += 1
        return Num(value=1)

    wf = WorkflowDefinition(id="w", summarize=summary, nodes=(
        TransformNode(id="a", fn=count_a),
        HumanApprovalNode(id="ok", depends_on=("a",), wait_kind="approval", build_request=const(1),
                          response_model=Approval),
        ToolNode(id="search", depends_on=("ok",), tool="search.web", permissions=frozenset({"network"}),
                 build_input=lambda v: {"query": str(v.output("ok", Approval).approved)}),
    ))
    first, task, _ = await run(wf)
    assert first.status == "waiting" and first.wait.kind == "approval" and first.wait.prompt == {"value": 1}
    again, task, _ = await run(wf, task)  # still no input: still waiting, nothing re-runs
    assert again.status == "waiting" and counter["a"] == 1
    task.workflow.node_states["ok"].human_input = {"approved": True}
    done, task, _ = await run(wf, task)
    assert done.status == "completed" and counter["a"] == 1
    assert task.workflow.node_states["search"].output == {"result": "True"}


async def test_pause_and_cancel_are_honoured_between_nodes() -> None:
    wf = WorkflowDefinition(id="w", summarize=summary, nodes=(TransformNode(id="a", fn=const(1)),))
    hooks = Hooks()
    hooks.control_state = TaskControl(pause_requested=True)
    assert (await run(wf, hooks=hooks))[0].status == "paused"
    hooks.control_state = TaskControl(cancel_requested=True)
    assert (await run(wf, hooks=hooks))[0].status == "cancelled"


def test_definition_validation() -> None:
    with pytest.raises(WorkflowDefinitionError, match="unknown node"):
        WorkflowDefinition(id="w", summarize=summary, nodes=(TransformNode(id="a", fn=const(1), depends_on=("zz",)),))
    with pytest.raises(WorkflowDefinitionError, match="duplicate"):
        WorkflowDefinition(id="w", summarize=summary, nodes=(TransformNode(id="a", fn=const(1)),
                                                              TransformNode(id="a", fn=const(1))))
    with pytest.raises(WorkflowDefinitionError, match="cycle"):
        WorkflowDefinition(id="w", summarize=summary, nodes=(TransformNode(id="a", fn=const(1), depends_on=("b",)),
                                                              TransformNode(id="b", fn=const(1), depends_on=("a",))))
