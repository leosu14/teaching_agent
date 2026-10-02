"""Execution scope: identifies where work runs (task/node/agent) and carries events and cost."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from app.observability.events import EventBus
from app.schemas.common import CostSummary, TokenUsage
from app.schemas.events import Event


class UsageLedger:
    """Accumulates token usage and cost for one task. The engine persists it with every checkpoint."""

    def __init__(self, summary: CostSummary | None = None) -> None:
        self.summary = summary.model_copy(deep=True) if summary else CostSummary()

    def record(self, *, agent_id: str, model: str, usage: TokenUsage, cost_usd: float) -> None:
        self.summary.record(agent_id=agent_id, model=model, usage=usage, cost_usd=cost_usd)

    def record_service(self, *, service: str, results: int, cache_hit: bool = False,
                       cost_usd: float | None = None, units: dict[str, float] | None = None,
                       estimated_cost_usd: float | None = None) -> None:
        self.summary.record_service(service=service, results=results, cache_hit=cache_hit, cost_usd=cost_usd,
                                    units=units, estimated_cost_usd=estimated_cost_usd)


@dataclass(frozen=True)
class ExecutionScope:
    events: EventBus
    usage: UsageLedger
    task_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None

    def for_node(self, node_id: str) -> ExecutionScope:
        return replace(self, node_id=node_id, agent_id=None)

    def for_agent(self, agent_id: str) -> ExecutionScope:
        return replace(self, agent_id=agent_id)

    def emit(self, type: str, *, tool: str | None = None, **data: object) -> Event:
        return self.events.emit(
            type, task_id=self.task_id, node_id=self.node_id, agent_id=self.agent_id, tool=tool, **data
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
