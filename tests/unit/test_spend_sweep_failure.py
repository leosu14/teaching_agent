"""MVP item 3 (spend backlog F7): a failing abandoned-reservation sweep must not block an unrelated paid call.

`SpendAccount.reserve` sweeps abandoned reservations before it reserves. The sweep only tidies up other attempts, so a
storage error in it is logged and the reservation goes ahead; the next reservation sweeps again. The reservation's own
transaction still decides alone whether the call may be sent: a full budget or a failing reserve still stops it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest

from app.observability.budget import BudgetExceededError
from app.schemas.common import utcnow
from app.schemas.events import EventType
from app.schemas.providers import Capability
from app.schemas.spend import SpendState
from app.schemas.usage import TaskBudget
from tests.unit.test_spend_invoker import ESTIMATE, call, counters, entries, setup, usage_of


class Broken:
    """Stands in for the store's sweep: raises (a storage hiccup) while `failing`, else runs the real sweep."""

    def __init__(self, store, failing: int = 10**6) -> None:
        self.real, self.failing, self.calls = store.expire_abandoned, failing, 0
        store.expire_abandoned = self

    def __call__(self, now):
        self.calls += 1
        if self.calls <= self.failing:
            raise RuntimeError("storage hiccup during the sweep")
        return self.real(now)


def counting(sent: list):
    async def fn():
        sent.append(1)
        return "answer"
    return fn


async def test_a_failing_sweep_does_not_block_an_unrelated_paid_call(caplog) -> None:
    inv, scope, ledger, seen = setup()
    sweep = Broken(ledger.account.store)
    sent: list = []
    with caplog.at_level(logging.WARNING, logger="app.observability.spend"):
        assert await call(inv, scope, counting(sent)) == "answer"
    assert sent == [1] and sweep.calls == 1
    [e] = entries(ledger)
    assert (e.state, e.usage_source) == (SpendState.SETTLED, "reported")  # reserved, sent and settled as usual
    assert counters(ledger) == (0, 50)
    assert ledger.account.swept == [] and not [ev for ev in seen if ev.type == EventType.SPEND_ABANDONED]
    [warning] = [r for r in caplog.records if "sweep failed" in r.getMessage()]
    assert warning.levelno == logging.WARNING and warning.exc_info is not None


async def test_the_next_reservation_sweeps_what_a_failed_sweep_left() -> None:
    inv, scope, ledger, seen = setup()
    store = ledger.account.store
    clock = {"now": utcnow()}
    ledger.account.clock = lambda: clock["now"]
    stuck = ledger.reserve(entry_id="preq_stuck", operation_id="pop_stuck", attempt=1, capability=Capability.LLM,
                           operation="generate", provider="vendor", model="m", estimate=ESTIMATE, timeout_seconds=1)
    assert stuck is not None
    clock["now"] = clock["now"] + timedelta(minutes=5)  # the stuck reservation is now abandoned
    Broken(store, failing=1)
    await call(inv, scope, counting([]))  # its sweep fails: the stuck entry is left reserved, the call goes ahead
    assert {e.entry_id: e.state for e in entries(ledger)}["preq_stuck"] == SpendState.RESERVED
    await call(inv, scope, counting([]))  # this one's sweep works and charges the abandoned estimate
    swept = {e.entry_id: e for e in entries(ledger)}["preq_stuck"]
    assert (swept.state, swept.error_type) == (SpendState.UNKNOWN, "Abandoned")
    assert [ev.data["entry_id"] for ev in seen if ev.type == EventType.SPEND_ABANDONED] == ["preq_stuck"]
    assert counters(ledger) == (0, 50 + 50 + 500)  # two reported calls and the abandoned one's estimate, once


async def test_a_failing_sweep_does_not_hide_a_full_budget() -> None:
    inv, scope, ledger, seen = setup(TaskBudget(max_llm_tokens=800))
    sent: list = []
    await call(inv, scope, counting(sent), usage=usage_of(300, 200))  # 500 settled
    Broken(ledger.account.store)
    with pytest.raises(BudgetExceededError):
        await call(inv, scope, counting(sent))  # 500 + 500 > 800: refused, whatever the sweep did
    assert sent == [1] and len(entries(ledger)) == 1
    assert [ev for ev in seen if ev.type == EventType.BUDGET_EXHAUSTED]


async def test_a_failing_reservation_still_stops_the_call() -> None:
    """Only the sweep is best-effort: when the reservation itself cannot be stored, nothing is sent."""
    inv, scope, ledger, _ = setup()
    store = ledger.account.store

    def broken_reserve(*_a, **_k):
        raise RuntimeError("storage down")

    store.reserve = broken_reserve
    sent: list = []
    with pytest.raises(RuntimeError, match="storage down"):
        await call(inv, scope, counting(sent))
    assert sent == [] and entries(ledger) == []


def test_a_cancellation_during_the_sweep_is_not_swallowed() -> None:
    """Only errors are absorbed: a cancellation (or KeyboardInterrupt) of the caller still propagates."""
    _, _, ledger, _ = setup()

    def cancelled(_now):
        raise asyncio.CancelledError()

    ledger.account.store.expire_abandoned = cancelled
    with pytest.raises(asyncio.CancelledError):
        ledger.reserve(entry_id="preq_x", operation_id="pop_x", attempt=1, capability=Capability.LLM, operation="generate",
                       provider="vendor", model="m", estimate=ESTIMATE, timeout_seconds=1)
    assert entries(ledger) == []

