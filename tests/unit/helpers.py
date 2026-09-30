from __future__ import annotations

from datetime import datetime, timezone

from app.config.routing import ModelPricing, RoutingConfig, RoutingTarget
from app.learner.frameworks import default_frameworks
from app.learner.memory import LearnerMemoryService
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.schemas.common import ModelTier
from app.schemas.events import Event
from app.schemas.learner import LearnerProfile

NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


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


class MemoryRepo:
    def __init__(self) -> None:
        self.rows: dict[str, LearnerProfile] = {}

    def get(self, learner_id):
        row = self.rows.get(learner_id)
        return row.model_copy(deep=True) if row else None

    def save(self, profile):
        self.rows[profile.learner_id] = profile.model_copy(deep=True)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self):
        return self.now


def service():
    clock = Clock()
    return LearnerMemoryService(MemoryRepo(), default_frameworks(), clock), clock
