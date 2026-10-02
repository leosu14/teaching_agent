"""Provider-layer schemas shared by configuration, providers, services and the API: capabilities, the per-call usage
record, health status and the call policy (timeout, retry, rate limit). Nothing here is vendor-specific."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.schemas.common import Schema


class Capability(str, Enum):
    LLM = "llm"
    TTS = "tts"
    IMAGE = "image"  # image generation: every image it returns has origin "generated"
    IMAGE_SEARCH = "image_search"  # images found by a search API: origin "searched" or "external"
    SEARCH = "search"  # web search for research


class ProviderUsage(Schema):
    """What one provider invocation used. A field the provider did not report stays None: values are never guessed.
    `estimated_cost` comes from configured pricing; `actual_cost` only from the provider itself."""

    provider: str
    capability: Capability
    operation: str
    model: str | None = None
    request_id: str  # ours, unique per invocation (also sent to the vendor where its API accepts one)
    vendor_request_id: str | None = None  # the vendor's id for the same request, when it returns one
    attempt: int = Field(default=1, ge=1)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    characters: int | None = None
    audio_seconds: float | None = None
    image_count: int | None = None
    result_count: int | None = None
    estimated_cost: float | None = None
    actual_cost: float | None = None
    currency: str | None = None
    latency_ms: float | None = None


class HealthStatus(Schema):
    provider: str
    capability: Capability
    available: bool
    latency_ms: float | None = None
    error: str | None = None
    checked: str = "local"  # "local": no network call was needed; "network": a cheap authenticated request


class ProviderPolicy(Schema):
    """How every call to one capability's providers is run. Retries are bounded; delays grow exponentially."""

    timeout_seconds: float = Field(default=60.0, gt=0, le=3600)  # per attempt
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_seconds: float = Field(default=0.5, ge=0, le=60)
    backoff_multiplier: float = Field(default=2.0, ge=1, le=10)
    max_backoff_seconds: float = Field(default=8.0, ge=0, le=300)
    requests_per_minute: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)

    def delay_before(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before `attempt` (1-based; the first attempt never waits). A provider's Retry-After is honoured up
        to `max_backoff_seconds`."""
        if attempt <= 1:
            return 0.0
        delay = self.backoff_seconds * self.backoff_multiplier ** (attempt - 2)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, self.max_backoff_seconds)

    def deadline_seconds(self) -> float:
        """Upper bound of one invocation including every retry and backoff."""
        return self.timeout_seconds * self.max_attempts + sum(self.delay_before(a) for a in
                                                              range(2, self.max_attempts + 1))
