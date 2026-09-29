from __future__ import annotations

from app.config.routing import ModelPricing, RoutingConfig, RoutingTarget
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.schemas.common import ModelTier
from app.schemas.events import Event


def routing(*chain: tuple[str, str]) -> RoutingConfig:
    chain = chain or (("mock", "m1"),)
    targets = [RoutingTarget(provider=p, model=m) for p, m in chain]
    return RoutingConfig(
        tiers={t: targets for t in ModelTier},
        pricing={m: ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0) for _, m in chain},
    )


def scope(task_id: str | None = "t1") -> tuple[ExecutionScope, list[Event]]:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    return ExecutionScope(events=bus, usage=UsageLedger(), task_id=task_id), seen
