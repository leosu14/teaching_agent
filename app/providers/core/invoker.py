"""ProviderInvoker: the one path every provider call takes. Per invocation it

- refuses network providers in offline mode,
- assigns a unique request id (in the usage record, the events and, through the adapters, the vendor request),
- applies the capability's local rate limit and per-attempt timeout,
- retries timeouts, rate limits and transient unavailability with bounded exponential backoff, nothing else,
- emits provider.request_started / request_completed / request_failed / rate_limited events, redacted,
- records a ProviderUsage per successful call.

Events go to the scope of the task being executed when there is one, otherwise to the application event bus.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.observability.events import EventBus
from app.observability.redaction import redact, redact_text
from app.observability.scope import current_scope
from app.providers.core.base import Provider, bind_request_id, new_request_id, release_request_id
from app.providers.core.errors import (
    ProviderError,
    ProviderOfflineError,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderUnavailable,
    is_retryable,
)
from app.providers.core.ratelimit import RateLimiter
from app.schemas.events import EventType
from app.schemas.providers import Capability, ProviderPolicy, ProviderUsage

T = TypeVar("T")
UsageOf = Callable[[T, str], ProviderUsage]  # (result, request id) -> usage
USAGE_LOG_SIZE = 1000


class ProviderInvoker:
    def __init__(self, *, policies: dict[Capability, ProviderPolicy] | None = None, events: EventBus | None = None,
                 offline: bool = False, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._policies = dict(policies or {})
        self._events = events
        self.offline = offline
        self._clock = clock
        self._sleep = sleep
        self._limiters: dict[tuple[Capability, str], RateLimiter] = {}
        self.usage_log: deque[ProviderUsage] = deque(maxlen=USAGE_LOG_SIZE)

    def policy(self, capability: Capability) -> ProviderPolicy:
        return self._policies.get(capability) or ProviderPolicy()

    def limiter(self, capability: Capability, provider: str) -> RateLimiter:
        key = (capability, provider)
        if key not in self._limiters:
            policy = self.policy(capability)
            self._limiters[key] = RateLimiter(requests_per_minute=policy.requests_per_minute,
                                              max_concurrency=policy.max_concurrency, sleep=self._sleep)
        return self._limiters[key]

    def emit(self, type: str, **data: object) -> None:
        data = redact(data)
        scope = current_scope()
        if scope is not None:
            scope.emit(type, **data)
        elif self._events is not None:
            self._events.emit(type, **data)

    def check_offline(self, provider: Provider) -> None:
        if self.offline and provider.requires_network:
            raise ProviderOfflineError(f"provider '{provider.name}' needs the network, but offline mode is on "
                                       "(TEACHING_AGENT_OFFLINE=true); only mock providers may run",
                                       provider=provider.name)

    async def call(self, provider: Provider, capability: Capability, operation: str,
                   fn: Callable[[], Awaitable[T]], *, usage: UsageOf, model: str | None = None) -> T:
        self.check_offline(provider)
        policy = self.policy(capability)
        limiter = self.limiter(capability, provider.name)
        base = {"provider": provider.name, "capability": capability.value, "operation": operation, "model": model}

        attempt = 0
        while True:
            attempt += 1
            request_id = new_request_id()

            def on_wait(reason: str, seconds: float | None) -> None:
                self.emit(EventType.PROVIDER_RATE_LIMITED, **base, request_id=request_id, source="local",
                          reason=reason, wait_seconds=seconds)

            self.emit(EventType.PROVIDER_REQUEST_STARTED, **base, request_id=request_id, attempt=attempt)
            token = bind_request_id(request_id)
            started = self._clock()
            try:
                async with limiter.slot(on_wait):
                    started = self._clock()
                    result = await asyncio.wait_for(fn(), policy.timeout_seconds)
            except Exception as raw:  # noqa: BLE001 - every failure is recorded, then re-raised typed
                exc = self._typed(raw, provider.name, policy)
                exc_request_id = getattr(exc, "request_id", None) or request_id
                if isinstance(exc, ProviderError):
                    exc.request_id = exc_request_id
                    exc.provider = exc.provider or provider.name
                retry = is_retryable(exc) and attempt < policy.max_attempts
                self.emit(EventType.PROVIDER_REQUEST_FAILED, **base, request_id=request_id, attempt=attempt,
                          error_type=type(exc).__name__, error=redact_text(str(exc))[:500],
                          status=getattr(exc, "status", None), will_retry=retry,
                          latency_ms=round((self._clock() - started) * 1000, 3))
                if isinstance(exc, ProviderRateLimit):
                    self.emit(EventType.PROVIDER_RATE_LIMITED, **base, request_id=request_id, source="provider",
                              reason="http_429", wait_seconds=exc.retry_after)
                if not retry:
                    if exc is raw:
                        raise
                    raise exc from raw
                await self._sleep(policy.delay_before(attempt + 1, getattr(exc, "retry_after", None)))
                continue
            finally:
                release_request_id(token)
            latency = round((self._clock() - started) * 1000, 3)
            record = usage(result, request_id).model_copy(update={"attempt": attempt, "latency_ms": latency})
            self.usage_log.append(record)
            self.emit(EventType.PROVIDER_REQUEST_COMPLETED, **base, request_id=request_id, attempt=attempt,
                      latency_ms=latency, usage=record.model_dump(mode="json", exclude_none=True))
            return result

    @staticmethod
    def _typed(exc: Exception, provider: str, policy: ProviderPolicy) -> Exception:
        """Map what is not yet a ProviderError onto the typed errors. Anything else (a programming error) is kept."""
        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, TimeoutError):
            return ProviderTimeout(f"no answer from '{provider}' within {policy.timeout_seconds}s", provider=provider)
        if isinstance(exc, (ConnectionError, OSError)):
            return ProviderUnavailable(f"'{provider}' unreachable: {exc}", provider=provider)
        return exc

    def fallback(self, capability: Capability, operation: str, *, from_provider: str, to_provider: str,
                 error: BaseException) -> None:
        self.emit(EventType.PROVIDER_FALLBACK, capability=capability.value, operation=operation,
                  from_provider=from_provider, to_provider=to_provider, error_type=type(error).__name__,
                  error=redact_text(str(error))[:500], request_id=getattr(error, "request_id", None))
