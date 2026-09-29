"""ModelRouter: the only path from agents to LLM providers. Handles tiers, fallback, timeouts and cost."""

from __future__ import annotations

import asyncio
import time

from app.config.routing import RoutingConfig, RoutingTarget
from app.observability.scope import ExecutionScope
from app.providers.llm.base import LLMMessage, LLMProvider, LLMRequest, LLMResponse, ProviderError
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

    def targets(self, tier: ModelTier) -> list[RoutingTarget]:
        return list(self._config.tiers[tier])

    def max_output_tokens(self, tier: ModelTier, requested: int) -> int:
        cap = self._config.max_output_tokens.get(tier)
        return min(requested, cap) if cap else requested

    def cost(self, model: str, usage: TokenUsage) -> float:
        price = self._config.pricing[model]
        uncached = usage.input_tokens - usage.cached_input_tokens
        return (
            uncached * price.input_per_mtok
            + usage.cached_input_tokens * price.cached_input_per_mtok
            + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000

    def estimate(self, tier: ModelTier, input_tokens: int, output_tokens: int) -> float:
        model = self._config.tiers[tier][0].model
        return self.cost(model, TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens))

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
        for target in self.targets(tier):
            request = LLMRequest(
                model=target.model,
                system=system,
                messages=messages,
                response_schema=response_schema,
                max_output_tokens=self.max_output_tokens(tier, max_output_tokens),
                agent_id=agent_id,
                attempt=attempt,
                input_payload=input_payload,
            )
            started = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._providers[target.provider].generate(request), timeout_seconds
                )
            except (ProviderError, TimeoutError) as exc:
                if isinstance(exc, ProviderError) and not exc.transient:
                    raise
                failures.append(f"{target.provider}/{target.model}: {exc!r}")
                scope.emit(EventType.LLM_FAILED, provider=target.provider, model=target.model, error=repr(exc))
                continue
            cost = self.cost(target.model, response.usage)
            scope.usage.record(agent_id=agent_id, model=target.model, usage=response.usage, cost_usd=cost)
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
