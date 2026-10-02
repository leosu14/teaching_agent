"""ModelRouter: the only path from agents to LLM providers. Handles tiers and per-agent routes, explicit fallback
chains, timeouts and cost. Providers are normally ManagedLLMProviders, so every call also gets the provider layer's
request id, retry, rate limit, events and usage record."""

from __future__ import annotations

import asyncio
import time

from app.config.routing import RoutingConfig, RoutingTarget
from app.observability.scope import ExecutionScope, active_scope
from app.providers.core.errors import ProviderError
from app.providers.llm.base import LLMMessage, LLMProvider, LLMRequest, LLMResponse
from app.schemas.common import ModelTier, TokenUsage
from app.schemas.events import EventType


class AllProvidersFailed(Exception):
    """Every target in the tier's fallback chain failed."""


class ModelRouter:
    def __init__(self, config: RoutingConfig, providers: dict[str, LLMProvider]) -> None:
        config.validate_against(set(providers))
        self._config = config
        self._providers = providers

    @property
    def provider_names(self) -> list[str]:
        return sorted(self._providers)

    def tier_for(self, agent_id: str, default: ModelTier) -> ModelTier:
        return self._config.agent_tiers.get(agent_id, default)

    @property
    def config(self) -> RoutingConfig:
        return self._config

    def targets(self, tier: ModelTier, agent_id: str | None = None) -> list[RoutingTarget]:
        return self._config.targets_for(agent_id, tier)

    def max_output_tokens(self, tier: ModelTier, requested: int) -> int:
        cap = self._config.max_output_tokens.get(tier)
        return min(requested, cap) if cap else requested

    def cost(self, model: str, usage: TokenUsage) -> float | None:
        """Cost in USD from configured pricing; None for a model configured without a price (unknown, not zero)."""
        return self._config.cost(model, usage)

    def estimate(self, tier: ModelTier, input_tokens: int, output_tokens: int) -> float:
        model = self._config.tiers[tier][0].model
        return self.cost(model, TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)) or 0.0

    async def complete(
        self,
        *,
        tier: ModelTier,
        agent_id: str,
        system: str,
        messages: list[LLMMessage],
        response_schema: dict | None,
        max_output_tokens: int,
        input_payload: dict,
        scope: ExecutionScope,
        timeout_seconds: float,
        attempt: int = 1,
    ) -> LLMResponse:
        failures: list[str] = []
        targets = self.targets(tier, agent_id)
        for index, target in enumerate(targets):
            if index:
                previous = targets[index - 1]
                scope.emit(EventType.PROVIDER_FALLBACK, capability="llm", operation="generate",
                           from_provider=previous.provider, from_model=previous.model, to_provider=target.provider,
                           to_model=target.model, error=failures[-1][:500])
            request = LLMRequest(
                model=target.model,
                system=system,
                messages=messages,
                response_schema=response_schema,
                max_output_tokens=self.max_output_tokens(tier, max_output_tokens),
                temperature=self._config.temperature,
                agent_id=agent_id,
                attempt=attempt,
                input_payload=input_payload,
            )
            provider = self._providers[target.provider]
            # The agent's LLM timeout bounds the call, unless the provider's own policy (per-attempt timeout times
            # bounded retries) needs longer: then the provider layer's timeouts are the bound.
            deadline = max(timeout_seconds, getattr(provider, "deadline_seconds", 0.0))
            started = time.perf_counter()
            try:
                with active_scope(scope):
                    response = await asyncio.wait_for(provider.generate(request), deadline)
            except (ProviderError, TimeoutError) as exc:
                if isinstance(exc, ProviderError) and not exc.transient:
                    raise
                failures.append(f"{target.provider}/{target.model}: {exc!r}")
                scope.emit(EventType.LLM_FAILED, provider=target.provider, model=target.model, error=repr(exc))
                continue
            cost = self.cost(target.model, response.usage)
            scope.usage.record(agent_id=agent_id, model=target.model, usage=response.usage, cost_usd=cost or 0.0)
            scope.emit(
                EventType.LLM_CALL,
                provider=target.provider,
                model=target.model,
                tier=tier.value,
                attempt=attempt,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=cost,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            return response
        raise AllProvidersFailed(f"all providers failed for tier {tier.value}: {failures}")
