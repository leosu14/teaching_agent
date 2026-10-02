"""ProviderSelector: decides which provider serves a capability, from configuration, the registry and known health.

Agents never pick vendors. The composition root asks the selector for each capability's chain (primary, then the
explicitly configured fallback); LLM selection follows the routing config (agent route, else the agent's tier).
A provider known to be unavailable is skipped only when a fallback is configured; otherwise the primary is kept,
so a failure surfaces instead of being silently masked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.routing import RoutingConfig
from app.providers.core.base import Provider
from app.providers.core.registry import ProviderNotFound, ProviderRegistry
from app.schemas.common import ModelTier
from app.schemas.providers import Capability


@dataclass(frozen=True)
class ProviderSelection:
    capability: Capability
    provider: str
    model: str | None
    fallbacks: tuple[str, ...] = ()
    fallback_models: tuple[str | None, ...] = ()
    available: bool | None = None  # last known health; None when never checked
    reason: str = "configured"
    skipped: tuple[str, ...] = field(default_factory=tuple)  # providers passed over because they were unavailable


class ProviderSelector:
    def __init__(self, registry: ProviderRegistry, *, chains: dict[Capability, list[str]] | None = None,
                 models: dict[Capability, str | None] | None = None, routing: RoutingConfig | None = None) -> None:
        self._registry = registry
        self._chains = {cap: list(chain) for cap, chain in (chains or {}).items() if chain}
        self._models = dict(models or {})
        self._routing = routing

    def configured_chain(self, capability: Capability) -> list[str]:
        chain = self._chains.get(capability)
        if chain:
            return list(chain)
        default = self._registry.default_id(capability)
        return [default] if default else []

    def select(self, capability: Capability, *, provider_id: str | None = None, model: str | None = None,
               agent_id: str | None = None, tier: ModelTier = ModelTier.STANDARD) -> ProviderSelection:
        """The provider (and model) for a capability. `provider_id` forces one provider; for the LLM, `agent_id` and
        `tier` select the agent's route."""
        if capability == Capability.LLM and self._routing is not None and provider_id is None:
            targets = self._routing.targets_for(agent_id, tier)
            ids = [t.provider for t in targets]
            models = [t.model for t in targets]
            reason = "agent route" if agent_id and self._routing.routes.get(agent_id) else f"tier '{tier.value}'"
        else:
            ids = [provider_id] if provider_id else self.configured_chain(capability)
            models = [model or self._models.get(capability)] + [None] * (len(ids) - 1)
            reason = "requested" if provider_id else "configured"
        if not ids:
            raise ProviderNotFound(f"no provider is configured for '{capability.value}'")
        for pid in ids:
            self._registry.get(capability, pid)  # every link must be registered: fail at startup, not mid-task

        chosen, skipped = 0, []
        if len(ids) > 1:
            for index, pid in enumerate(ids):
                health = self._registry.last_health(capability, pid)
                if health is not None and not health.available and index < len(ids) - 1:
                    skipped.append(pid)
                    continue
                chosen = index
                break
            if skipped:
                reason = f"{reason}; skipped unavailable {', '.join(skipped)}"
        health = self._registry.last_health(capability, ids[chosen])
        return ProviderSelection(
            capability=capability, provider=ids[chosen], model=models[chosen], fallbacks=tuple(ids[chosen + 1:]),
            fallback_models=tuple(models[chosen + 1:]), available=None if health is None else health.available,
            reason=reason, skipped=tuple(skipped),
        )

    def chain(self, capability: Capability) -> list[Provider]:
        """Provider instances in the order they are tried: the selected one, then its configured fallbacks."""
        selection = self.select(capability)
        return [self._registry.get(capability, pid) for pid in (selection.provider, *selection.fallbacks)]

    def describe(self) -> list[ProviderSelection]:
        return [self.select(cap) for cap in Capability if self.configured_chain(cap) or
                (cap == Capability.LLM and self._routing is not None)]
