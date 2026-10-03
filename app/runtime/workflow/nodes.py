"""Workflow node types. Nodes describe WHAT runs; the engine decides when, retries, times out and checkpoints."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, TypeVar

from pydantic import BaseModel

from app.agents.base import AgentContext
from app.agents.registry import AgentRegistry
from app.observability.scope import ExecutionScope
from app.providers.llm.router import ModelRouter
from app.schemas.common import RetryPolicy
from app.schemas.events import EventType
from app.schemas.lesson import ReviewResult, Verdict
from app.schemas.task import Task, WaitRequest
from app.schemas.workflow import (
    BranchDecision,
    NodeState,
    NodeStatus,
    ParallelOutcome,
    ReviewOutcome,
    ReviewRound,
    RevisionPolicy,
    RevisionRequest,
    WorkflowState,
)
from app.tools.base import ToolCaller
from app.tools.manager import ToolManager

M = TypeVar("M", bound=BaseModel)


class NodeOutputMissing(LookupError):
    pass


class NodeFatal(Exception):
    """A node failed in a way that retrying the node cannot fix."""


class ReviewRejected(NodeFatal):
    pass


class StateView:
    """Read-only access to completed node outputs, validated back into their schemas."""

    def __init__(self, state: WorkflowState, task: Task) -> None:
        self._state = state
        self.task = task

    def status(self, node_id: str) -> NodeStatus:
        ns = self._state.node_states.get(node_id)
        return ns.status if ns else NodeStatus.PENDING

    def output(self, node_id: str, model: type[M]) -> M:
        ns = self._state.node_states.get(node_id)
        if ns is None or ns.status != NodeStatus.COMPLETED or ns.output is None:
            raise NodeOutputMissing(f"node '{node_id}' has no output")
        return model.model_validate(ns.output)

    def maybe(self, node_id: str, model: type[M]) -> M | None:
        return self.output(node_id, model) if self.status(node_id) == NodeStatus.COMPLETED else None

    def finished_at(self, node_id: str) -> datetime | None:
        ns = self._state.node_states.get(node_id)
        return ns.finished_at if ns else None


@dataclass
class NodeRuntime:
    agents: AgentRegistry
    tools: ToolManager
    router: ModelRouter
    scope: ExecutionScope
    view: StateView
    node_state: NodeState
    checkpoint: Callable[[], None]
    run_child: Callable[[Node], Awaitable[NodeState]]
    # Whether the task's owner asked to cancel. The engine checks between nodes; long-waiting nodes check it too.
    cancel_requested: Callable[[], bool] = lambda: False

    def agent_context(self) -> AgentContext:
        return AgentContext(router=self.router, tools=self.tools, scope=self.scope)


@dataclass(frozen=True)
class NodeResult:
    output: BaseModel | None = None
    wait: WaitRequest | None = None
    skip: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Node(ABC):
    id: str
    depends_on: tuple[str, ...] = ()
    after: tuple[str, ...] = ()  # ordering-only dependencies: a skipped `after` node does not skip this one
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_seconds: float | None = None
    kind: ClassVar[str] = "node"

    @abstractmethod
    async def execute(self, rt: NodeRuntime) -> NodeResult: ...

    async def on_cancel(self, rt: NodeRuntime) -> None:
        """Release outside work (e.g. provider jobs) when the task is cancelled while this node waits. Default:
        nothing to release."""


InputBuilder = Callable[[StateView], BaseModel | dict]


@dataclass(frozen=True, kw_only=True)
class AgentNode(Node):
    agent: str
    build_input: InputBuilder
    retry: RetryPolicy = field(default_factory=lambda: RetryPolicy(max_attempts=2))
    kind: ClassVar[str] = "agent"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        agent = rt.agents.get(self.agent)
        return NodeResult(output=await agent.execute(self.build_input(rt.view), rt.agent_context()))


@dataclass(frozen=True, kw_only=True)
class ToolNode(Node):
    tool: str
    build_input: InputBuilder
    permissions: frozenset[str] = frozenset()
    kind: ClassVar[str] = "tool"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        caller = ToolCaller(caller_id=f"node:{self.id}", allowed_tools=frozenset({self.tool}),
                            permissions=self.permissions)
        return NodeResult(output=await rt.tools.call(caller, self.tool, self.build_input(rt.view), rt.scope))


@dataclass(frozen=True, kw_only=True)
class TransformNode(Node):
    fn: Callable[[StateView], BaseModel]
    kind: ClassVar[str] = "transform"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        return NodeResult(output=self.fn(rt.view))


@dataclass(frozen=True, kw_only=True)
class ConditionalNode(Node):
    predicate: Callable[[StateView], bool]
    when_true: tuple[str, ...] = ()
    when_false: tuple[str, ...] = ()
    kind: ClassVar[str] = "conditional"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        branch = bool(self.predicate(rt.view))
        skipped = self.when_false if branch else self.when_true
        return NodeResult(output=BranchDecision(branch=branch, skipped=list(skipped)), skip=skipped)


@dataclass(frozen=True, kw_only=True)
class ParallelNode(Node):
    children: tuple[Node, ...]
    kind: ClassVar[str] = "parallel"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        states = await asyncio.gather(*(rt.run_child(child) for child in self.children), return_exceptions=True)
        errors = [s for s in states if isinstance(s, BaseException)]
        if errors:
            raise errors[0]
        return NodeResult(output=ParallelOutcome(children=[c.id for c in self.children]))


@dataclass(frozen=True, kw_only=True)
class HumanApprovalNode(Node):
    """Pauses the task in WAITING until a person supplies input matching `response_model`.

    `validate_input` adds checks that need workflow state (raise ValueError to reject). `wait_event` is
    emitted when the task starts waiting here and `submitted_event` when valid input is accepted.
    """

    wait_kind: str
    build_request: Callable[[StateView], BaseModel]
    response_model: type[BaseModel]
    validate_input: Callable[[StateView, BaseModel], None] | None = None
    wait_event: str | None = None
    submitted_event: str | None = None
    kind: ClassVar[str] = "human"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        if rt.node_state.human_input is None:
            prompt = self.build_request(rt.view).model_dump(mode="json")
            return NodeResult(wait=WaitRequest(node_id=self.id, kind=self.wait_kind, prompt=prompt))
        return NodeResult(output=self.response_model.model_validate(rt.node_state.human_input))


@dataclass(frozen=True, kw_only=True)
class ReviewNode(Node):
    """Generate -> review -> revise loop with issue propagation, a revision budget and an exhaustion policy.

    Progress is checkpointed after every generation and every review, so a resumed task continues the
    loop where it stopped instead of starting over.
    """

    generator: str
    reviewer: str
    candidate_model: type[BaseModel]
    build_generator_input: Callable[[StateView, RevisionRequest | None], BaseModel]
    build_reviewer_input: Callable[[StateView, BaseModel, int], BaseModel]
    policy: RevisionPolicy = field(default_factory=RevisionPolicy)
    kind: ClassVar[str] = "review"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        progress = rt.node_state.progress
        rounds = [ReviewRound.model_validate(r) for r in progress.get("rounds", [])]
        generator = rt.agents.get(self.generator)
        reviewer = rt.agents.get(self.reviewer)
        actx = rt.agent_context()

        while True:
            if rounds and rounds[-1].verdict == Verdict.APPROVED:
                return NodeResult(output=self._outcome("approved", progress, rounds))
            if rounds and len(rounds) - 1 >= self.policy.max_revisions:
                if self.policy.on_exhausted == "accept_with_warnings":
                    return NodeResult(output=self._outcome("accepted_with_warnings", progress, rounds))
                raise ReviewRejected(
                    f"review still requires revision after {self.policy.max_revisions} revisions: "
                    + "; ".join(i.problem for i in rounds[-1].issues)
                )

            revision_number = len(rounds)
            if progress.get("candidate_for") != revision_number:
                revision = None
                if revision_number:
                    revision = RevisionRequest(revision_number=revision_number, issues=rounds[-1].issues,
                                               previous=progress["candidate"])
                candidate = await generator.execute(self.build_generator_input(rt.view, revision), actx)
                progress["candidate"] = candidate.model_dump(mode="json")
                progress["candidate_for"] = revision_number
                rt.checkpoint()

            candidate = self.candidate_model.model_validate(progress["candidate"])
            rt.scope.emit(EventType.REVIEW_STARTED, revision=revision_number)
            review = await reviewer.execute(self.build_reviewer_input(rt.view, candidate, revision_number), actx)
            assert isinstance(review, ReviewResult)
            rounds.append(ReviewRound(revision_number=revision_number, verdict=review.verdict,
                                      scores={k.value: v for k, v in review.scores.items()}, issues=review.issues))
            progress["rounds"] = [r.model_dump(mode="json") for r in rounds]
            progress["last_review"] = review.model_dump(mode="json")
            rt.checkpoint()
            rt.scope.emit(EventType.REVIEW_COMPLETED, revision=revision_number, verdict=review.verdict.value,
                          issues=len(review.issues))

    def _outcome(self, status: str, progress: dict, rounds: list[ReviewRound]) -> ReviewOutcome:
        return ReviewOutcome(
            status=status, candidate=progress["candidate"], candidate_type=self.candidate_model.__name__,
            revisions=len(rounds) - 1, rounds=rounds,
            final_review=ReviewResult.model_validate(progress["last_review"]),
        )
