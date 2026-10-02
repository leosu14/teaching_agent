from __future__ import annotations

import pytest

from app.config.routing import ConfigError
from app.providers.core.errors import ProviderError
from app.providers.llm.base import LLMMessage
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.router import AllProvidersFailed, ModelRouter
from app.schemas.common import ModelTier, TokenUsage
from tests.unit.helpers import routing, scope


def echo(_request):
    return {"ok": True}


async def complete(router, sc, tier=ModelTier.CHEAP):
    return await router.complete(tier=tier, agent_id="a", system="s", messages=[LLMMessage(role="user", content="hi")],
                                 response_schema=None, max_output_tokens=100, input_payload={}, scope=sc,
                                 timeout_seconds=1)


async def test_routes_records_usage_and_cost() -> None:
    router = ModelRouter(routing(("mock", "m1")), {"mock": MockLLMProvider({"a": echo})})
    sc, events = scope()
    response = await complete(router, sc)
    assert response.model == "m1" and response.text == '{"ok": true}'
    summary = sc.usage.summary
    assert summary.llm_calls == 1 and summary.actual_cost_usd == pytest.approx(router.cost("m1", response.usage))
    assert summary.by_agent["a"].calls == 1 and summary.by_model["m1"].calls == 1
    assert [e.type for e in events] == ["llm.call"]


async def test_falls_back_to_next_target_on_transient_error() -> None:
    primary = MockLLMProvider({"a": echo}, name="p1")
    primary.inject("a", ProviderError("rate limited"))
    backup = MockLLMProvider({"a": echo}, name="p2")
    router = ModelRouter(routing(("p1", "m1"), ("p2", "m2")), {"p1": primary, "p2": backup})
    sc, events = scope()
    response = await complete(router, sc)
    assert response.provider == "p2"
    assert [e.type for e in events] == ["llm.failed", "provider.fallback", "llm.call"]  # fallback is recorded


async def test_non_transient_error_is_not_masked_by_fallback() -> None:
    primary = MockLLMProvider({"a": echo}, name="p1")
    primary.inject("a", ProviderError("bad request", transient=False))
    router = ModelRouter(routing(("p1", "m1"), ("p2", "m2")), {"p1": primary, "p2": MockLLMProvider({"a": echo}, name="p2")})
    with pytest.raises(ProviderError):
        await complete(router, scope()[0])


async def test_all_targets_failing_raises() -> None:
    p = MockLLMProvider({"a": echo})
    p.inject("a", ProviderError("down"))
    router = ModelRouter(routing(("mock", "m1")), {"mock": p})
    with pytest.raises(AllProvidersFailed):
        await complete(router, scope()[0])


def test_cost_uses_cached_pricing_and_estimates() -> None:
    config = routing(("mock", "m1"))
    config.pricing["m1"].cached_input_per_mtok = 0.5
    router = ModelRouter(config, {"mock": MockLLMProvider({})})
    usage = TokenUsage(input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=1_000_000)
    assert router.cost("m1", usage) == pytest.approx(0.5 + 0.25 + 2.0)
    assert router.estimate(ModelTier.CHEAP, 1_000_000, 0) == pytest.approx(1.0)


def test_tier_overrides_and_config_validation() -> None:
    config = routing(("mock", "m1"))
    config.agent_tiers["teacher"] = ModelTier.CHEAP
    router = ModelRouter(config, {"mock": MockLLMProvider({})})
    assert router.tier_for("teacher", ModelTier.REASONING) == ModelTier.CHEAP
    assert router.tier_for("planner", ModelTier.REASONING) == ModelTier.REASONING
    with pytest.raises(ConfigError, match="not enabled"):
        ModelRouter(routing(("anthropic", "m1")), {"mock": MockLLMProvider({})})
    bad = routing(("mock", "m1"))
    bad.pricing.clear()
    with pytest.raises(ConfigError, match="no pricing"):
        ModelRouter(bad, {"mock": MockLLMProvider({})})
