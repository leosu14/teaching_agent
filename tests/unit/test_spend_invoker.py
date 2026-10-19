"""Reservation and settlement around each provider attempt (PR19): what the ledger keeps for a successful attempt,
one without reported usage, a timeout, a cancellation, an unusable answer, a refused request and an attempt the
budget has no room for."""

from __future__ import annotations

import asyncio

import pytest

from app.observability.budget import BudgetExceededError
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.providers.core.errors import ProviderAuthenticationError, ProviderResponseError, ProviderTimeout
from app.providers.core.invoker import ProviderInvoker
from app.schemas.events import Event, EventType
from app.schemas.providers import Capability, ProviderPolicy, ProviderUsage
from app.schemas.spend import Estimate, SpendState
from app.schemas.usage import TaskBudget


class Vendor:
    name = "vendor"
    requires_network = False


async def no_sleep(_seconds: float) -> None:
    return None


def setup(budget: TaskBudget | None = None, *, attempts: int = 1, timeout: float = 5.0):
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    ledger = UsageLedger(budget=budget)
    scope = ExecutionScope(events=bus, usage=ledger, task_id="t1")
    policy = ProviderPolicy(max_attempts=attempts, timeout_seconds=timeout, backoff_seconds=0)
    return ProviderInvoker(policies={c: policy for c in Capability}, sleep=no_sleep), scope, ledger, seen


def usage_of(input_tokens: int | None, output_tokens: int | None):
    def of(_result, request_id: str) -> ProviderUsage:
        return ProviderUsage(provider="vendor", capability=Capability.LLM, operation="generate", model="m",
                             request_id=request_id, input_tokens=input_tokens, output_tokens=output_tokens)
    return of


ESTIMATE = Estimate(llm_tokens=500)


async def call(inv, scope, fn, *, usage=usage_of(30, 20), estimate=ESTIMATE):
    with active_scope(scope):
        return await inv.call(Vendor(), Capability.LLM, "generate", fn, usage=usage, model="m", estimate=estimate)


def entries(ledger: UsageLedger):
    return ledger.account.store.entries()


def counters(ledger: UsageLedger, metric: str = "llm_tokens"):
    [key] = ledger.account.scope_keys
    t = ledger.account.store.totals([key]).get(key, {}).get(metric)
    return (t.reserved, t.settled) if t else (0, 0)


async def ok():
    return "answer"


async def test_reported_usage_replaces_the_estimate() -> None:
    inv, scope, ledger, _ = setup()
    assert await call(inv, scope, ok) == "answer"
    [e] = entries(ledger)
    assert (e.state, e.usage_source, e.charged["llm_tokens"], e.estimate["llm_tokens"]) == (
        SpendState.SETTLED, "reported", 50, 500)
    assert e.entry_id.startswith("preq_") and e.operation_id.startswith("pop_")  # our ids, not the vendor's
    assert counters(ledger) == (0, 50)


async def test_missing_usage_keeps_the_estimate_and_says_so() -> None:
    inv, scope, ledger, _ = setup()
    await call(inv, scope, ok, usage=usage_of(None, None))
    [e] = entries(ledger)
    assert (e.state, e.usage_source, e.charged["llm_tokens"], e.reported) == (SpendState.SETTLED, "estimate", 500, None)


async def test_a_timeout_may_have_been_billed_so_its_estimate_is_charged() -> None:
    inv, scope, ledger, seen = setup(timeout=0.01)

    async def slow():
        await asyncio.sleep(1)

    with pytest.raises(Exception):
        await call(inv, scope, slow)
    [e] = entries(ledger)
    assert (e.state, e.usage_source, e.charged["llm_tokens"]) == (SpendState.UNKNOWN, "estimate", 500)
    unknown = [ev for ev in seen if ev.type == EventType.SPEND_UNKNOWN]
    assert unknown and unknown[0].data["reason"] == "timeout"


async def test_an_unusable_answer_is_unknown_and_a_refusal_charges_only_the_request() -> None:
    inv, scope, ledger, _ = setup()

    async def garbled():
        raise ProviderResponseError("not json", provider="vendor")

    async def refused():
        raise ProviderAuthenticationError("bad key", provider="vendor")

    for fn in (garbled, refused):
        with pytest.raises(Exception):
            await call(inv, scope, fn)
    unknown, failed = entries(ledger)
    assert unknown.state == SpendState.UNKNOWN and unknown.charged["llm_tokens"] == 500
    assert failed.state == SpendState.FAILED and failed.charged == {"llm_requests": 1}
    assert counters(ledger) == (0, 500) and counters(ledger, "llm_requests") == (0, 2)


async def test_cancellation_after_sending_is_unknown_and_before_sending_is_released() -> None:
    inv, scope, ledger, _ = setup()
    started = asyncio.Event()

    async def hangs():
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(call(inv, scope, hangs))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):  # never swallowed
        await task
    [sent] = entries(ledger)
    assert (sent.state, sent.error_type, sent.charged["llm_tokens"]) == (SpendState.UNKNOWN, "CancelledError", 500)

    # Cancelled while waiting for a concurrency slot: nothing reached the provider.
    inv2, scope2, ledger2, _ = setup()
    inv2._policies = {c: ProviderPolicy(max_attempts=1, max_concurrency=1) for c in Capability}
    first = asyncio.create_task(call(inv2, scope2, hangs))
    await asyncio.sleep(0)
    while len(entries(ledger2)) < 1:
        await asyncio.sleep(0)
    second = asyncio.create_task(call(inv2, scope2, hangs))
    while len(entries(ledger2)) < 2:
        await asyncio.sleep(0)
    second.cancel()
    first.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    states = {e.state for e in entries(ledger2)}
    assert states == {SpendState.UNKNOWN, SpendState.RELEASED}
    assert counters(ledger2) == (0, 500)


async def test_a_request_the_budget_has_no_room_for_is_never_sent() -> None:
    inv, scope, ledger, seen = setup(TaskBudget(max_llm_tokens=800))
    sent = []

    async def counted():
        sent.append(1)
        return "answer"

    await call(inv, scope, counted, usage=usage_of(300, 200))  # 500 reserved, 500 settled
    with pytest.raises(BudgetExceededError) as raised:
        await call(inv, scope, counted)  # 500 + 500 > 800
    assert "MAX_LLM_TOKENS" in str(raised.value) and len(sent) == 1
    exhausted = [ev for ev in seen if ev.type == EventType.BUDGET_EXHAUSTED]
    assert exhausted and exhausted[0].data["budget"] == "task"
    assert len(entries(ledger)) == 1


async def test_retries_are_separate_attempts_of_one_operation() -> None:
    inv, scope, ledger, _ = setup(attempts=3)
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ProviderTimeout("slow", provider="vendor")
        return "answer"

    await call(inv, scope, flaky)
    es = entries(ledger)
    assert [e.attempt for e in es] == [1, 2, 3] and len({e.operation_id for e in es}) == 1
    assert len({e.entry_id for e in es}) == 3
    assert [e.state for e in es] == [SpendState.UNKNOWN, SpendState.UNKNOWN, SpendState.SETTLED]
