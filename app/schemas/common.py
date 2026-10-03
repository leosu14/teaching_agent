"""Shared primitives used by every layer: ids, time, retry policy, token usage and cost."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

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
    """Usage of a non-LLM service (search, retrieval, images, speech). Costs stay None unless the provider reports
    them: `cost_usd` is the actual cost, `estimated_cost_usd` a provider's estimate (it is not added to the task's
    actual cost)."""

    calls: int = 0
    cache_hits: int = 0
    results: int = 0
    cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    units: dict[str, float] = Field(default_factory=dict)  # provider-reported units, e.g. images, characters, seconds


class ProviderRequestRecord(Schema):
    """One provider request (one attempt) made on behalf of a task, with where in the workflow it was made. Fields
    the provider did not report stay None."""

    request_id: str
    provider: str
    capability: str  # a Capability value: llm, tts, image, image_search, search, video_generation
    operation: str
    model: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    attempt: int = 1
    status: Literal["ok", "failed"] = "ok"
    error_type: str | None = None
    vendor_request_id: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    characters: int | None = None
    audio_seconds: float | None = None
    image_count: int | None = None
    result_count: int | None = None
    video_seconds: float | None = None  # generated video seconds
    estimated_cost_usd: float | None = None  # from configured pricing
    actual_cost_usd: float | None = None  # reported by the provider
    at: datetime = Field(default_factory=utcnow)


class CostSummary(Schema):
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    llm_calls: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    by_agent: dict[str, CostLine] = Field(default_factory=dict)
    by_model: dict[str, CostLine] = Field(default_factory=dict)
    by_service: dict[str, ServiceUsageLine] = Field(default_factory=dict)
    provider_requests: list[ProviderRequestRecord] = Field(default_factory=list)  # every attempt, in order

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
                       cost_usd: float | None = None, units: dict[str, float] | None = None,
                       estimated_cost_usd: float | None = None) -> None:
        line = self.by_service.setdefault(service, ServiceUsageLine())
        for unit, amount in (units or {}).items():
            line.units[unit] = round(line.units.get(unit, 0.0) + amount, 6)
        if cache_hit:
            line.cache_hits += 1
        else:
            line.calls += 1
        line.results += results
        if cost_usd is not None:
            line.cost_usd = round((line.cost_usd or 0.0) + cost_usd, 8)
            self.actual_cost_usd = round(self.actual_cost_usd + cost_usd, 8)
        if estimated_cost_usd is not None:
            line.estimated_cost_usd = round((line.estimated_cost_usd or 0.0) + estimated_cost_usd, 8)
