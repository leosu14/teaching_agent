"""ProviderInvoker: the one path every provider call takes. Per invocation it

- refuses network providers in offline mode,
- assigns a unique request id (in the usage record, the events and, through the adapters, the vendor request),
- applies the capability's local rate limit and per-attempt timeout,
- retries timeouts, rate limits and transient unavailability with bounded exponential backoff, nothing else,
- emits provider.request_started / request_completed / request_failed / rate_limited events, redacted,
- records a ProviderUsage per successful call,
- for a call made on behalf of a task, session or graded attempt: reserves every billable attempt against the work's
  budgets before it is sent (a request that would exceed one is never sent; a task worker that lost its lease cannot
  reserve) and settles it once its outcome is known, in the durable spend ledger, whatever happens to the lease
  afterwards. A timed-out, cancelled or garbled attempt that may have been billed is recorded as `unknown` with its
  estimate charged; every attempt is also kept in the work's in-memory usage summary.

Events go to the scope of the task being executed when there is one, otherwise to the application event bus.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.observability.budget import BudgetExceededError
from app.observability.events import EventBus
from app.observability.redaction import redact, redact_text
from app.observability.scope import UsageLedger, current_scope
from app.observability.spend import Reservation
from app.providers.core.base import Provider, bind_request_id, new_request_id, release_request_id
from app.providers.core.errors import (
    ProviderError,
    ProviderOfflineError,
    ProviderRateLimit,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
    is_retryable,
)
from app.providers.core.ratelimit import RateLimiter
from app.schemas.common import ProviderRequestRecord, new_id
from app.schemas.events import EventType
from app.schemas.providers import Capability, ProviderPolicy, ProviderUsage
from app.schemas.spend import Estimate, SpendState

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
                                       "(TEACHING_AGENT_MODE=offline or TEACHING_AGENT_OFFLINE=true); only mock providers "
                                       "may run",
                                       provider=provider.name)

    async def call(self, provider: Provider, capability: Capability, operation: str,
                   fn: Callable[[], Awaitable[T]], *, usage: UsageOf, model: str | None = None,
                   estimate: Estimate | None = None, idempotency_key: str | None = None) -> T:
        """Run one provider operation. `estimate` is what each attempt is expected to use, known before it is sent
        (e.g. the characters of a speech request, an LLM request's token bound): it is reserved against the budgets
        of the work's account before the attempt is sent, and replaced by what the provider reports afterwards.
        `idempotency_key` identifies a paid result (e.g. a video generation key): at most one attempt for it may be
        in flight."""
        self.check_offline(provider)
        policy = self.policy(capability)
        limiter = self.limiter(capability, provider.name)
        base = {"provider": provider.name, "capability": capability.value, "operation": operation, "model": model}
        scope = current_scope()
        ledger = scope.usage if scope is not None else None
        operation_id = new_id("pop")

        attempt = 0
        while True:
            attempt += 1
            request_id = new_request_id()
            reservation = None
            if ledger is not None:
                try:  # BudgetExceededError, LeaseLost, SpendInFlight: nothing is sent
                    reservation = ledger.reserve(
                        entry_id=request_id, operation_id=operation_id, attempt=attempt, capability=capability,
                        operation=operation, provider=provider.name, model=model, estimate=estimate or Estimate(),
                        timeout_seconds=policy.timeout_seconds, node_id=scope.node_id, agent_id=scope.agent_id,
                        idempotency_key=idempotency_key)
                except BudgetExceededError as exc:
                    self._spend_event(scope, EventType.BUDGET_EXHAUSTED, budget=exc.scope, limit=exc.limit,
                                      used=exc.used, maximum=exc.maximum, **base)
                    raise
                for entry_id in ledger.account.swept:
                    self._spend_event(scope, EventType.SPEND_ABANDONED, entry_id=entry_id)

            def on_wait(reason: str, seconds: float | None) -> None:
                self.emit(EventType.PROVIDER_RATE_LIMITED, **base, request_id=request_id, source="local",
                          reason=reason, wait_seconds=seconds)

            self.emit(EventType.PROVIDER_REQUEST_STARTED, **base, request_id=request_id, attempt=attempt)
            token = bind_request_id(request_id)
            started = self._clock()
            sent = False
            try:
                async with limiter.slot(on_wait):
                    started = self._clock()
                    sent = True
                    result = await asyncio.wait_for(fn(), policy.timeout_seconds)
            except asyncio.CancelledError:
                # Cancelled from outside (an agent, router or node timeout; a caller that went away). Once the request
                # was sent the provider may bill it: recorded as unknown, its estimate charged. Never swallowed.
                if ledger is not None:
                    record = None
                    if sent:
                        record = ProviderRequestRecord(
                            request_id=request_id, provider=provider.name, capability=capability.value,
                            operation=operation, model=model, attempt=attempt, status="failed",
                            error_type="CancelledError", latency_ms=round((self._clock() - started) * 1000, 3),
                            node_id=scope.node_id, agent_id=scope.agent_id)
                    self._settle(ledger, scope, reservation, record,
                                 SpendState.UNKNOWN if sent else SpendState.RELEASED, "cancelled", base)
                raise
            except Exception as raw:  # noqa: BLE001 - every failure is recorded, then re-raised typed
                exc = self._typed(raw, provider.name, policy)
                exc_request_id = getattr(exc, "request_id", None) or request_id
                if isinstance(exc, ProviderError):
                    exc.request_id = exc_request_id
                    exc.provider = exc.provider or provider.name
                retry = is_retryable(exc) and attempt < policy.max_attempts
                latency = round((self._clock() - started) * 1000, 3)
                if ledger is not None:
                    state = failure_state(exc) if sent else SpendState.RELEASED
                    self._settle(ledger, scope, reservation, ProviderRequestRecord(
                        request_id=request_id, provider=provider.name, capability=capability.value,
                        operation=operation, model=model, attempt=attempt, status="failed",
                        error_type=type(exc).__name__, latency_ms=latency, node_id=scope.node_id,
                        agent_id=scope.agent_id), state, _unknown_reason(exc), base)
                self.emit(EventType.PROVIDER_REQUEST_FAILED, **base, request_id=request_id, attempt=attempt,
                          error_type=type(exc).__name__, error=redact_text(str(exc))[:500],
                          status=getattr(exc, "status", None), will_retry=retry, latency_ms=latency)
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
            if ledger is not None:  # BudgetExceededError when the reported usage is over a limit (still recorded)
                self._settle(ledger, scope, reservation, ProviderRequestRecord(
                    request_id=request_id, provider=record.provider, capability=capability.value,
                    operation=operation, model=record.model or model, attempt=attempt,
                    vendor_request_id=record.vendor_request_id, latency_ms=latency, input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens, characters=record.characters,
                    audio_seconds=record.audio_seconds, image_count=record.image_count,
                    result_count=record.result_count, video_seconds=record.video_seconds,
                    estimated_cost_usd=record.estimated_cost, actual_cost_usd=record.actual_cost,
                    node_id=scope.node_id, agent_id=scope.agent_id), SpendState.SETTLED, None, base)
            return result

    def _settle(self, ledger: UsageLedger, scope, reservation: Reservation | None,
                record: ProviderRequestRecord | None, state: SpendState, reason: str | None, base: dict) -> None:
        result = ledger.settle(reservation, record, state)
        if reservation is None:
            return
        entry_id = reservation.entry_id
        if result is None:
            self._spend_event(scope, EventType.SPEND_SETTLE_FAILED, entry_id=entry_id, state=state.value, **base)
            return
        if state == SpendState.UNKNOWN and result.outcome == "applied":
            self._spend_event(scope, EventType.SPEND_UNKNOWN, entry_id=entry_id, reason=reason,
                              estimate=reservation.amounts, **base)
        if result.outcome == "conflict":
            self._spend_event(scope, EventType.SPEND_SETTLE_CONFLICT, entry_id=entry_id, state=state.value, **base)
        if result.stale_lease:
            self._spend_event(scope, EventType.SPEND_AFTER_LEASE_LOSS, entry_id=entry_id, state=state.value,
                              stale_fence=scope.fence.fence if scope.fence else None, **base)

    def _spend_event(self, scope, type: str, **data: object) -> None:
        """Spend events are facts about money, stored even when the emitting worker has lost its lease (unfenced,
        like task.lease_lost). Ids and numbers only."""
        data = redact(data)
        bus = scope.events if scope is not None else self._events
        if bus is not None:
            bus.emit(type, task_id=scope.task_id if scope else None, node_id=scope.node_id if scope else None,
                     agent_id=scope.agent_id if scope else None, fence=None, **data)

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


def failure_state(exc: BaseException) -> SpendState:
    """The ledger state of a sent attempt that failed: `failed` when the provider refused or never processed it
    (only request counts are charged), `unknown` when it may have been billed (its estimate is charged)."""
    billed = getattr(exc, "possibly_billed", None)  # what the transport knows about the vendor's side, if anything
    if billed is not None:
        return SpendState.UNKNOWN if billed else SpendState.FAILED
    if isinstance(exc, (ProviderTimeout, TimeoutError, ProviderResponseError)):
        return SpendState.UNKNOWN
    if isinstance(exc, ProviderError):
        return SpendState.FAILED
    return SpendState.UNKNOWN  # an unexpected error after the request left: assume it was billed


def _unknown_reason(exc: BaseException) -> str:
    if isinstance(exc, (ProviderTimeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, ProviderResponseError):
        return "response_error"
    if getattr(exc, "possibly_billed", None):
        return "server_error" if getattr(exc, "status", None) else "connection_lost"
    return "unexpected_error"
