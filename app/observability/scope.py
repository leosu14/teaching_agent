"""Execution scope: identifies where work runs (task/node/agent) and carries events and cost."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from app.observability.events import EventBus
from app.observability.spend import Reservation, SpendAccount
from app.schemas.common import CostSummary, ProviderRequestRecord, TokenUsage
from app.schemas.events import Event
from app.schemas.lease import Lease
from app.schemas.spend import SettleResult, SpendState
from app.schemas.usage import TaskBudget, TaskUsage


class UsageLedger:
    """Accumulates token usage, cost and every provider request of one piece of work (the engine checkpoints it
    with the task), and spends against its account: each billable attempt is reserved before it is sent and settled
    after. With a durable account (task workers, teaching sessions, API grading) the spend ledger in the database is
    the source of truth for budgets; without one an in-memory account enforces `budget` the same way."""

    def __init__(self, summary: CostSummary | None = None, *, budget: TaskBudget | None = None,
                 account: SpendAccount | None = None) -> None:
        self.summary = summary.model_copy(deep=True) if summary else CostSummary()
        self.budget = budget
        self.account = account if account is not None else SpendAccount.local(
            budget, task_usage(self.summary) if summary else None)

    def usage(self) -> TaskUsage:
        return task_usage(self.summary)

    def reserve(self, **attempt: Any) -> Reservation | None:
        """Called by the provider layer before each attempt; raises BudgetExceededError (nothing is sent), or
        LeaseLost when the task worker's lease is gone. None for an operation that is not billed."""
        return self.account.reserve(**attempt)

    def settle(self, reservation: Reservation | None, record: ProviderRequestRecord | None,
               state: SpendState) -> SettleResult | None:
        """Called by the provider layer once an attempt's outcome is known (successful or not). Raises
        BudgetExceededError after recording a successful attempt whose reported usage crossed a limit."""
        if record is not None:
            self.summary.provider_requests.append(record)
        if reservation is None:
            return None
        result = self.account.settle(reservation, record, state)
        if result is not None and state == SpendState.SETTLED:
            self.account.check_after(reservation, result)
        return result

    def record(self, *, agent_id: str, model: str, usage: TokenUsage, cost_usd: float) -> None:
        self.summary.record(agent_id=agent_id, model=model, usage=usage, cost_usd=cost_usd)

    def record_service(self, *, service: str, results: int, cache_hit: bool = False,
                       cost_usd: float | None = None, units: dict[str, float] | None = None,
                       estimated_cost_usd: float | None = None) -> None:
        self.summary.record_service(service=service, results=results, cache_hit=cache_hit, cost_usd=cost_usd,
                                    units=units, estimated_cost_usd=estimated_cost_usd)


def task_usage(summary: CostSummary) -> TaskUsage:
    """TaskUsage from a task's cost summary: its provider requests plus the local video rendering time."""
    render = sum(line.units.get("render_seconds", 0.0) for name, line in summary.by_service.items()
                 if name.startswith("video_compose:"))
    return TaskUsage.from_records(summary.provider_requests, video_render_seconds=render)


@dataclass(frozen=True)
class ExecutionScope:
    events: EventBus
    usage: UsageLedger
    task_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    fence: Lease | None = None  # the task worker's lease: its events and artifacts are stored only while it holds
    # Raises LeaseLost when `fence` is no longer current. Called before writes outside the task's own records
    # (learner memory, assessments): a stale worker stops there instead of writing them.
    fence_check: Callable[[], None] | None = None

    def require_fence(self) -> None:
        if self.fence_check is not None:
            self.fence_check()

    def for_node(self, node_id: str) -> ExecutionScope:
        return replace(self, node_id=node_id, agent_id=None)

    def for_agent(self, agent_id: str) -> ExecutionScope:
        return replace(self, agent_id=agent_id)

    def emit(self, type: str, *, tool: str | None = None, **data: object) -> Event:
        return self.events.emit(
            type, task_id=self.task_id, node_id=self.node_id, agent_id=self.agent_id, tool=tool, fence=self.fence,
            **data
        )


# The scope of the work running in the current asyncio task. Tools and the model router set it, so code below them
# (provider calls) can attribute its events to the right task, node and agent without a scope parameter.
_current_scope: ContextVar[ExecutionScope | None] = ContextVar("current_scope", default=None)


def current_scope() -> ExecutionScope | None:
    return _current_scope.get()


@contextmanager
def active_scope(scope: ExecutionScope) -> Iterator[ExecutionScope]:
    token = _current_scope.set(scope)
    try:
        yield scope
    finally:
        _current_scope.reset(token)
