"""Model routing configuration: which provider/model serves each tier (or a specific agent), pricing and output
limits. A target list is an ordered, explicit fallback chain: the second target is used only when the first fails
transiently."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from app.schemas.common import ModelTier, Schema, TokenUsage


class ConfigError(ValueError):
    """Raised when configuration is missing or inconsistent. Messages say what to fix."""


class RoutingTarget(Schema):
    provider: str
    model: str


class ModelPricing(Schema):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cached_input_per_mtok: float = Field(default=0.0, ge=0)


class RoutingConfig(Schema):
    tiers: dict[ModelTier, list[RoutingTarget]]
    pricing: dict[str, ModelPricing]
    agent_tiers: dict[str, ModelTier] = Field(default_factory=dict)
    # Per-agent routes (e.g. the teacher or the reviewer on their own provider/model); they replace the tier chain.
    routes: dict[str, list[RoutingTarget]] = Field(default_factory=dict)
    max_output_tokens: dict[ModelTier, int] = Field(default_factory=dict)
    temperature: float | None = None  # None: the provider's default (some models accept no temperature at all)
    # Models configured from the environment without a price: their cost is reported as unknown, never guessed.
    unpriced_models: list[str] = Field(default_factory=list)

    def targets_for(self, agent_id: str | None, tier: ModelTier) -> list[RoutingTarget]:
        if agent_id is not None and self.routes.get(agent_id):
            return list(self.routes[agent_id])
        return list(self.tiers[tier])

    def cost(self, model: str, usage: TokenUsage) -> float | None:
        """USD cost from configured pricing; None for a model without a price (unknown, never guessed)."""
        price = self.pricing.get(model)
        if price is None:
            return None
        uncached = usage.input_tokens - usage.cached_input_tokens
        return (
            uncached * price.input_per_mtok
            + usage.cached_input_tokens * price.cached_input_per_mtok
            + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000

    def all_targets(self) -> list[RoutingTarget]:
        seen: dict[tuple[str, str], RoutingTarget] = {}
        for targets in [*self.tiers.values(), *self.routes.values()]:
            for t in targets:
                seen.setdefault((t.provider, t.model), t)
        return list(seen.values())

    def providers(self) -> list[str]:
        return sorted({t.provider for t in self.all_targets()})

    def validate_against(self, available_providers: set[str]) -> None:
        missing_tiers = [t.value for t in ModelTier if not self.tiers.get(t)]
        if missing_tiers:
            raise ConfigError(f"routing: no targets configured for tiers {missing_tiers}")
        chains = [(f"tier '{tier.value}'", targets) for tier, targets in self.tiers.items()]
        chains += [(f"agent route '{agent}'", targets) for agent, targets in self.routes.items()]
        for where, targets in chains:
            for target in targets:
                if target.provider not in available_providers:
                    raise ConfigError(
                        f"routing: {where} uses provider '{target.provider}', "
                        f"which is not enabled (enabled: {sorted(available_providers)})"
                    )
                if target.model not in self.pricing and target.model not in self.unpriced_models:
                    raise ConfigError(f"routing: model '{target.model}' has no pricing entry")


def load_routing(path: Path) -> RoutingConfig:
    if not path.exists():
        raise ConfigError(f"routing file not found: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    try:
        return RoutingConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"routing file {path} is invalid: {exc}") from exc
