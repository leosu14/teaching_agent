"""The spend ledger at the store boundary (PR19): reservation, settlement, idempotency, abandoned reservations and
lease fencing, for the SQL store (SQLite) and the in-memory one. A controllable clock: no timing, no sleeps."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.observability.spend import InMemorySpendStore, abandoned
from app.schemas.common import utcnow
from app.schemas.lease import LeaseLost
from app.schemas.spend import (
    Charge,
    ReservationRefused,
    SpendInFlight,
    SpendReservation,
    SpendSettlement,
    SpendState,
)
from app.schemas.task import Task
from app.storage.db import create_db
from app.storage.repositories import SqlSpendRepository, SqlTaskRepository

TTL = timedelta(minutes=5)


class Clock:
    def __init__(self) -> None:
        self.offset = timedelta(0)

    def __call__(self):
        return utcnow() + self.offset


@pytest.fixture(params=["sql", "memory"])
def ledger(request, tmp_path):
    clock = Clock()
    if request.param == "memory":
        return InMemorySpendStore(), None, clock
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    tasks = SqlTaskRepository(sessions, clock)
    assert tasks.create(Task(task_id="t1", user_id="u", learner_id="l", request="r"))
    return SqlSpendRepository(sessions, clock), tasks, clock


@pytest.fixture
def sql(tmp_path):
    clock = Clock()
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    tasks = SqlTaskRepository(sessions, clock)
    assert tasks.create(Task(task_id="t1", user_id="u", learner_id="l", request="r"))
    return SqlSpendRepository(sessions, clock), tasks, clock


def reservation(entry_id: str, *charges: Charge, clock=utcnow, key: str | None = None, attempt: int = 1,
                fence: int | None = None, worker: str | None = None, ttl: float = 60) -> SpendReservation:
    return SpendReservation(
        entry_id=entry_id, operation_id=f"pop_{entry_id}", attempt=attempt, idempotency_key=key, capability="llm",
        operation="generate", provider="p", task_id="t1", worker=worker, lease_fence=fence, charges=list(charges),
        estimate={c.metric: c.amount for c in charges}, estimated_cost_microusd=None,
        reserved_until=clock() + timedelta(seconds=ttl))


def tokens(amount: int, limit: int | None = 1000, scope: str = "task:t1") -> Charge:
    return Charge(scope_key=scope, metric="llm_tokens", amount=amount, check=max(amount, 1), limit=limit)


def settled(**charged: int) -> SpendSettlement:
    return SpendSettlement(state=SpendState.SETTLED, charged=charged, usage_source="reported")


def totals(store, scope: str = "task:t1", metric: str = "llm_tokens"):
    t = store.totals([scope]).get(scope, {}).get(metric)
    return (t.reserved, t.settled) if t else (0, 0)


def test_a_reservation_that_does_not_fit_is_refused_and_writes_nothing(ledger) -> None:
    store, _, clock = ledger
    store.reserve(reservation("a", tokens(600), clock=clock), fence=None, now=clock())
    with pytest.raises(ReservationRefused) as refused:
        store.reserve(reservation("b", tokens(600), clock=clock), fence=None, now=clock())
    assert (refused.value.reserved, refused.value.settled) == (600, 0)
    assert totals(store) == (600, 0) and [e.entry_id for e in store.entries(task_id="t1")] == ["a"]


def test_all_scopes_or_none(ledger) -> None:
    """A reservation charges every scope of its account in one transaction: one full scope refuses all of them."""
    store, _, clock = ledger
    store.reserve(reservation("a", tokens(90, limit=100, scope="global:d"), clock=clock), fence=None, now=clock())
    with pytest.raises(ReservationRefused):
        store.reserve(reservation("b", tokens(50), tokens(50, limit=100, scope="global:d"), clock=clock),
                      fence=None, now=clock())
    assert totals(store) == (0, 0) and totals(store, "global:d") == (90, 0)


def test_a_used_up_limit_refuses_even_a_zero_estimate(ledger) -> None:
    store, _, clock = ledger
    store.reserve(reservation("a", tokens(1000), clock=clock), fence=None, now=clock())
    store.settle("a", settled(llm_tokens=1000), now=clock())
    with pytest.raises(ReservationRefused):  # check = max(amount, 1): one unit of headroom is required
        store.reserve(reservation("b", tokens(0), clock=clock), fence=None, now=clock())


def test_settlement_replaces_the_estimate_and_is_idempotent(ledger) -> None:
    store, _, clock = ledger
    store.reserve(reservation("a", tokens(500), clock=clock), fence=None, now=clock())
    store.reserve(reservation("a", tokens(500), clock=clock), fence=None, now=clock())  # same id: no-op
    assert totals(store) == (500, 0)
    assert store.settle("a", settled(llm_tokens=120), now=clock()).outcome == "applied"
    assert totals(store) == (0, 120)
    assert store.settle("a", settled(llm_tokens=120), now=clock()).outcome == "duplicate"
    conflict = store.settle("a", SpendSettlement(state=SpendState.FAILED), now=clock())
    assert conflict.outcome == "conflict" and totals(store) == (0, 120)  # the first settlement stands
    [entry] = store.entries(task_id="t1")
    assert entry.state == SpendState.SETTLED and entry.charged == {"llm_tokens": 120}


def test_released_and_failed_keep_only_what_they_charge(ledger) -> None:
    store, _, clock = ledger
    requests = Charge(scope_key="task:t1", metric="llm_requests", amount=1, check=1, limit=10)
    store.reserve(reservation("a", requests, tokens(300), clock=clock), fence=None, now=clock())
    store.reserve(reservation("b", requests, tokens(300), clock=clock), fence=None, now=clock())
    store.settle("a", SpendSettlement(state=SpendState.RELEASED), now=clock())
    store.settle("b", SpendSettlement(state=SpendState.FAILED, charged={"llm_requests": 1}), now=clock())
    assert totals(store) == (0, 0) and totals(store, metric="llm_requests") == (0, 1)


def test_an_abandoned_reservation_is_charged_its_estimate_then_corrected_by_a_late_result(ledger) -> None:
    """A crashed process never settles: once `reserved_until` passes, the next sweep marks it unknown and charges its
    estimate (conservative, labelled an estimate). If the original process does answer later, its real usage
    replaces the estimate, once."""
    store, _, clock = ledger
    store.reserve(reservation("a", tokens(400), clock=clock, ttl=60), fence=None, now=clock())
    assert store.expire_abandoned(clock()) == []  # not expired yet
    clock.offset = timedelta(seconds=61)
    assert store.expire_abandoned(clock()) == ["a"]
    assert store.expire_abandoned(clock()) == []  # each entry is swept once
    [entry] = store.entries(task_id="t1")
    assert (entry.state, entry.usage_source, entry.error_type) == (SpendState.UNKNOWN, "estimate", "Abandoned")
    assert entry.final_cost_microusd is None and entry.cost_source == "unpriced"  # never presented as reported
    assert totals(store) == (0, 400)

    assert store.settle("a", settled(llm_tokens=150), now=clock()).outcome == "applied"
    assert totals(store) == (0, 150)
    assert store.settle("a", settled(llm_tokens=150), now=clock()).outcome == "duplicate"
    # A late failure cannot undo the conservative charge.
    store.reserve(reservation("b", tokens(100), clock=clock, ttl=0), fence=None, now=clock())
    clock.offset += timedelta(seconds=1)
    store.expire_abandoned(clock())
    assert store.settle("b", SpendSettlement(state=SpendState.RELEASED), now=clock()).outcome == "conflict"
    assert totals(store) == (0, 250)


def test_abandoned_settlement_marks_the_estimate_as_an_estimate() -> None:
    entry_reservation = reservation("a", tokens(10))
    store = InMemorySpendStore()
    store.reserve(entry_reservation.model_copy(update={"estimated_cost_microusd": 2500}), fence=None, now=utcnow())
    s = abandoned(store.entries()[0])
    assert (s.state, s.final_cost_microusd, s.cost_source, s.usage_source) == (
        SpendState.UNKNOWN, 2500, "estimate", "estimate")


def test_one_in_flight_attempt_per_idempotency_key(ledger) -> None:
    store, _, clock = ledger
    store.reserve(reservation("a", tokens(1), clock=clock, key="video:k"), fence=None, now=clock())
    with pytest.raises(SpendInFlight):
        store.reserve(reservation("b", tokens(1), clock=clock, key="video:k", attempt=2), fence=None, now=clock())
    assert totals(store) == (1, 0)  # the refused attempt reserved nothing
    store.settle("a", SpendSettlement(state=SpendState.FAILED), now=clock())
    store.reserve(reservation("b", tokens(1), clock=clock, key="video:k", attempt=2), fence=None, now=clock())


def test_seeding_imports_earlier_spend_once(ledger) -> None:
    store, _, clock = ledger
    assert store.seed("task:t1", {"llm_tokens": 700}, now=clock())
    assert not store.seed("task:t1", {"llm_tokens": 9999}, now=clock())
    assert totals(store) == (0, 700)
    with pytest.raises(ReservationRefused):
        store.reserve(reservation("a", tokens(400), clock=clock), fence=None, now=clock())


def test_a_reservation_is_fenced_by_the_task_lease_and_its_settlement_is_not(sql) -> None:
    store, tasks, clock = sql
    old = tasks.claim("t1", "old", TTL)
    store.reserve(reservation("a", tokens(100), clock=clock, fence=old.fence, worker="old"), fence=old, now=clock())
    clock.offset = TTL + timedelta(seconds=1)
    new = tasks.claim("t1", "new", TTL)
    with pytest.raises(LeaseLost):  # the stale worker cannot start another paid call
        store.reserve(reservation("b", tokens(1), clock=clock), fence=old, now=clock())
    late = store.settle("a", settled(llm_tokens=80), now=clock())  # but its call in flight is still recorded
    assert late.outcome == "applied" and late.stale_lease
    assert totals(store) == (0, 80)
    store.reserve(reservation("c", tokens(100), clock=clock, fence=new.fence, worker="new"), fence=new, now=clock())
    assert not store.settle("c", settled(llm_tokens=10), now=clock()).stale_lease
    assert totals(store) == (0, 90)


def test_counters_always_equal_the_sum_of_their_entries(sql) -> None:
    """Invariant: per scope and metric, `reserved` is the estimate of the entries still reserved and `settled` the
    charge of the others, whatever mix of outcomes, duplicates and sweeps happened."""
    store, _, clock = sql
    outcomes = [settled(llm_tokens=50), SpendSettlement(state=SpendState.FAILED), None,
                SpendSettlement(state=SpendState.RELEASED), SpendSettlement(state=SpendState.UNKNOWN,
                                                                            charged={"llm_tokens": 70})]
    for i, outcome in enumerate(outcomes * 3):
        store.reserve(reservation(f"e{i}", tokens(70, limit=None), clock=clock, ttl=30 if i % 2 else 600),
                      fence=None, now=clock())
        if outcome is not None:
            store.settle(f"e{i}", outcome, now=clock())
            store.settle(f"e{i}", outcome, now=clock())
    clock.offset = timedelta(seconds=31)
    store.expire_abandoned(clock())
    entries = store.entries(task_id="t1")
    reserved = sum(e.estimate.get("llm_tokens", 0) for e in entries if e.state == SpendState.RESERVED)
    charged = sum(e.charged.get("llm_tokens", 0) for e in entries if e.state != SpendState.RESERVED)
    assert totals(store) == (reserved, charged)
    assert any(e.state == SpendState.RESERVED for e in entries) and any(e.state == SpendState.UNKNOWN for e in entries)
