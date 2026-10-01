"""Shared primitives used by every layer: ids, time, retry policy, token usage and cost."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Schema(BaseModel):
    """Base for every schema: unknown fields are rejected so malformed LLM output fails loudly."""

    model_config = ConfigDict(extra="forbid")


class ModelTier(str, Enum):
    REASONING = "reasoning"
    STANDARD = "standard"
    CHEAP = "cheap"


class RetryPolicy(Schema):
    max_attempts: int = Field(default=1, ge=1)
    backoff_seconds: float = Field(default=0.0, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)

    def delay_before(self, attempt: int) -> float:
        """Delay before `attempt` (1-based). The first attempt never waits."""
        if attempt <= 1:
            return 0.0
        return self.backoff_seconds * self.backoff_multiplier ** (attempt - 2)


class TokenUsage(Schema):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def plus(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


class CostLine(Schema):
    calls: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0


class ServiceUsageLine(Schema):
    """Usage of a non-LLM service (search, retrieval). `cost_usd` stays None unless the provider reports a cost."""

    calls: int = 0
    cache_hits: int = 0
    results: int = 0
    cost_usd: float | None = None


class CostSummary(Schema):
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    llm_calls: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    by_agent: dict[str, CostLine] = Field(default_factory=dict)
    by_model: dict[str, CostLine] = Field(default_factory=dict)
    by_service: dict[str, ServiceUsageLine] = Field(default_factory=dict)

    def record(self, *, agent_id: str, model: str, usage: TokenUsage, cost_usd: float) -> None:
        self.llm_calls += 1
        self.actual_cost_usd = round(self.actual_cost_usd + cost_usd, 8)
        self.token_usage = self.token_usage.plus(usage)
        for key, table in ((agent_id, self.by_agent), (model, self.by_model)):
            line = table.setdefault(key, CostLine())
            line.calls += 1
            line.usage = line.usage.plus(usage)
            line.cost_usd = round(line.cost_usd + cost_usd, 8)

    def record_service(self, *, service: str, results: int, cache_hit: bool = False,
                       cost_usd: float | None = None) -> None:
        line = self.by_service.setdefault(service, ServiceUsageLine())
        if cache_hit:
            line.cache_hits += 1
        else:
            line.calls += 1
        line.results += results
        if cost_usd is not None:
            line.cost_usd = round((line.cost_usd or 0.0) + cost_usd, 8)
            self.actual_cost_usd = round(self.actual_cost_usd + cost_usd, 8)
