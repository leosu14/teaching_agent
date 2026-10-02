"""What every provider declares, whatever its capability: an id, its capabilities, whether it needs the network,
its (non-secret) configuration and a cheap health check."""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import ClassVar

from app.observability.redaction import redact_text
from app.providers.core.errors import ProviderError
from app.schemas.providers import Capability, HealthStatus

# The request id of the provider invocation running in this asyncio task, so an HTTP adapter can send it to the
# vendor without a parameter on every interface method.
_request_id: ContextVar[str | None] = ContextVar("provider_request_id", default=None)


def new_request_id() -> str:
    return f"preq_{uuid.uuid4().hex}"


def current_request_id() -> str | None:
    return _request_id.get()


def bind_request_id(request_id: str):
    return _request_id.set(request_id)


def release_request_id(token) -> None:
    _request_id.reset(token)


class Provider:
    """Mixin of every capability interface. `name` is the provider id (e.g. "mock", "openai")."""

    name: str
    capabilities: ClassVar[frozenset[Capability]] = frozenset()
    requires_network: ClassVar[bool] = False

    @property
    def provider_id(self) -> str:
        return self.name

    def configuration(self) -> dict:
        """Non-secret settings, for diagnostics. Implementations never include credentials."""
        return {}

    async def probe(self) -> str:
        """The cheapest possible check that the provider can serve requests. Raise ProviderError when it cannot.
        Returns "local" when no network request was made, "network" when one was."""
        return "local"

    async def health_check(self, capability: Capability | None = None) -> HealthStatus:
        capability = capability or next(iter(sorted(self.capabilities, key=lambda c: c.value)))
        started = time.perf_counter()
        try:
            checked = await self.probe()
        except ProviderError as exc:
            return HealthStatus(provider=self.name, capability=capability, available=False,
                                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                                error=redact_text(f"{type(exc).__name__}: {exc}"))
        return HealthStatus(provider=self.name, capability=capability, available=True, checked=checked,
                            latency_ms=round((time.perf_counter() - started) * 1000, 3))
