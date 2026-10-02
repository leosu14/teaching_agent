"""ProviderRegistry: every configured provider instance, by capability and id, with defaults and last health."""

from __future__ import annotations

import asyncio
import time

from app.config.routing import ConfigError
from app.providers.core.base import Provider
from app.providers.core.errors import ProviderOfflineError
from app.schemas.providers import Capability, HealthStatus


class ProviderNotFound(ConfigError):
    """No provider with that id is registered for that capability."""


class ProviderRegistry:
    def __init__(self, *, offline: bool = False) -> None:
        self.offline = offline
        self._providers: dict[Capability, dict[str, Provider]] = {}
        self._defaults: dict[Capability, str] = {}
        self._health: dict[tuple[Capability, str], HealthStatus] = {}

    def register(self, provider: Provider, *, capability: Capability | None = None, default: bool = False) -> None:
        """Register `provider` for `capability` (default: every capability it declares). The first provider of a
        capability becomes its default unless another is registered with default=True."""
        capabilities = [capability] if capability else sorted(provider.capabilities, key=lambda c: c.value)
        if not capabilities:
            raise ConfigError(f"provider '{provider.name}' declares no capabilities")
        if self.offline and provider.requires_network:
            raise ProviderOfflineError(f"provider '{provider.name}' needs the network, but offline mode is on "
                                       "(TEACHING_AGENT_OFFLINE=true); only mock providers may be registered",
                                       provider=provider.name)
        for cap in capabilities:
            if cap not in provider.capabilities:
                raise ConfigError(f"provider '{provider.name}' does not provide capability '{cap.value}'")
            existing = self._providers.setdefault(cap, {})
            if provider.name in existing and existing[provider.name] is not provider:
                raise ConfigError(f"a different '{provider.name}' provider is already registered for '{cap.value}'")
            existing[provider.name] = provider
            if default or cap not in self._defaults:
                self._defaults[cap] = provider.name

    def get(self, capability: Capability, provider_id: str) -> Provider:
        provider = self._providers.get(capability, {}).get(provider_id)
        if provider is None:
            raise ProviderNotFound(f"no '{capability.value}' provider '{provider_id}' is configured "
                                   f"(configured: {self.ids(capability)})")
        return provider

    def has(self, capability: Capability, provider_id: str) -> bool:
        return provider_id in self._providers.get(capability, {})

    def ids(self, capability: Capability) -> list[str]:
        return sorted(self._providers.get(capability, {}))

    def providers(self, capability: Capability) -> list[Provider]:
        return [self._providers[capability][pid] for pid in self.ids(capability)]

    def capabilities(self) -> dict[Capability, list[str]]:
        """Capability discovery: which provider ids serve each capability."""
        return {cap: self.ids(cap) for cap in sorted(self._providers, key=lambda c: c.value)}

    def set_default(self, capability: Capability, provider_id: str) -> None:
        self.get(capability, provider_id)
        self._defaults[capability] = provider_id

    def default(self, capability: Capability) -> Provider:
        provider_id = self._defaults.get(capability)
        if provider_id is None:
            raise ProviderNotFound(f"no provider is configured for '{capability.value}'")
        return self.get(capability, provider_id)

    def default_id(self, capability: Capability) -> str | None:
        return self._defaults.get(capability)

    def last_health(self, capability: Capability, provider_id: str) -> HealthStatus | None:
        return self._health.get((capability, provider_id))

    def record_health(self, status: HealthStatus) -> None:
        self._health[(status.capability, status.provider)] = status

    async def check_health(self, capability: Capability | None = None, *,
                           timeout_seconds: float = 10.0) -> list[HealthStatus]:
        """Run every (or one capability's) provider health check, concurrently and bounded by `timeout_seconds`."""
        targets = [(cap, p) for cap in sorted(self._providers, key=lambda c: c.value)
                   if capability in (None, cap) for p in self.providers(cap)]

        async def check(cap: Capability, provider: Provider) -> HealthStatus:
            if self.offline and provider.requires_network:
                return HealthStatus(provider=provider.name, capability=cap, available=False,
                                    error="offline mode: network providers are disabled")
            started = time.perf_counter()
            try:
                return await asyncio.wait_for(provider.health_check(cap), timeout_seconds)
            except TimeoutError:
                return HealthStatus(provider=provider.name, capability=cap, available=False,
                                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                                    error=f"health check timed out after {timeout_seconds}s")

        results = await asyncio.gather(*(check(cap, p) for cap, p in targets))
        for status in results:
            self.record_health(status)
        return list(results)
