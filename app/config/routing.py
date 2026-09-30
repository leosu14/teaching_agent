"""Model routing configuration: which provider/model serves each tier, pricing and output limits."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from app.schemas.common import ModelTier, Schema


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
    max_output_tokens: dict[ModelTier, int] = Field(default_factory=dict)

    def validate_against(self, available_providers: set[str]) -> None:
        missing_tiers = [t.value for t in ModelTier if not self.tiers.get(t)]
        if missing_tiers:
            raise ConfigError(f"routing: no targets configured for tiers {missing_tiers}")
        for tier, targets in self.tiers.items():
            for target in targets:
                if target.provider not in available_providers:
                    raise ConfigError(
                        f"routing: tier '{tier.value}' uses provider '{target.provider}', "
                        f"which is not enabled (enabled: {sorted(available_providers)})"
                    )
                if target.model not in self.pricing:
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
