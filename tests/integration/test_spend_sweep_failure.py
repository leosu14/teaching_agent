"""MVP item 3 (spend backlog F7) on the SQL spend ledger: a failing abandoned-reservation sweep never blocks paid work.

The sweep runs inline before every reservation. If it hits a storage error, the reservation goes ahead and the next
one sweeps again. Within one sweep, each abandoned entry is settled in its own transaction, so one entry that cannot
be settled is left reserved for the next sweep and does not stop the others.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from app.observability.spend import task_account
from app.schemas.common import new_id, utcnow
from app.schemas.providers import Capability
from app.schemas.spend import Charge, Estimate, SpendReservation, SpendState
from app.schemas.task import TaskStatus
from app.schemas.usage import BudgetDefaults
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlSpendRepository
from tests.conftest import run_lesson


def reservation(entry_id: str, scope: str, *, ttl: float = 1, amount: int = 100) -> SpendReservation:
    charge = Charge(scope_key=scope, metric="llm_tokens", amount=amount, check=amount, limit=None)
    return SpendReservation(entry_id=entry_id, operation_id=new_id("pop"), attempt=1, capability="llm",
                            operation="generate", provider="p", task_id=scope.split(":", 1)[1], charges=[charge],
                            estimate={"llm_tokens": amount}, reserved_until=utcnow() + timedelta(seconds=ttl))


def totals(store: SqlSpendRepository, scope: str) -> tuple[int, int]:
    t = store.totals([scope])[scope]["llm_tokens"]
    return t.reserved, t.settled


def fail_settling(store: SqlSpendRepository, entry_id: str) -> None:
    """Settling `entry_id` raises (a storage error on that one entry); every other settlement works."""
    real = store.settle

    def settle(eid, settlement, *, now):
        if eid == entry_id:
            raise RuntimeError(f"storage error settling {eid}")
        return real(eid, settlement, now=now)

    store.settle = settle


def test_one_entry_that_cannot_be_settled_does_not_stop_the_sweep(tmp_path, caplog) -> None:
    sessions = create_db(f"sqlite:///{tmp_path / 'spend.db'}")
    try:
        store = SqlSpendRepository(sessions)
        scope = f"task:{new_id('task')}"
        ids = [new_id("preq") for _ in range(3)]
        for entry_id in ids:
            store.reserve(reservation(entry_id, scope), fence=None, now=utcnow())
        later = utcnow() + timedelta(seconds=5)
        fail_settling(store, ids[1])
        with caplog.at_level(logging.WARNING, logger="app.storage.repositories"):
            assert sorted(store.expire_abandoned(later)) == sorted([ids[0], ids[2]])  # the others are still swept
        assert any(ids[1] in r.getMessage() for r in caplog.records)
        states = {e.entry_id: e.state for e in store.entries(task_id=scope.split(":", 1)[1])}
        assert states == {ids[0]: SpendState.UNKNOWN, ids[1]: SpendState.RESERVED, ids[2]: SpendState.UNKNOWN}
        assert totals(store, scope) == (100, 200)

        del store.settle  # the storage recovers: the next sweep settles what was left, once
        assert store.expire_abandoned(later) == [ids[1]]
        assert store.expire_abandoned(later) == []
        assert totals(store, scope) == (0, 300)
    finally:
        dispose(sessions)


def test_a_reservation_goes_ahead_when_the_sweep_fails_and_the_next_one_sweeps(tmp_path) -> None:
    sessions = create_db(f"sqlite:///{tmp_path / 'spend.db'}")
    try:
        store = SqlSpendRepository(sessions)
        task_id = new_id("task")
        stuck = new_id("preq")
        store.reserve(reservation(stuck, f"task:{task_id}"), fence=None, now=utcnow())
        later = utcnow() + timedelta(seconds=5)
        account = task_account(store, BudgetDefaults(), task_id=task_id, learner_id=None, budget=None, fence=None,
                               clock=lambda: later)
        fail_settling(store, stuck)

        def reserve(entry_id: str):
            return account.reserve(entry_id=entry_id, operation_id=new_id("pop"), attempt=1,
                                   capability=Capability.LLM, operation="generate", provider="p",
                                   model="m", estimate=Estimate(llm_tokens=100), timeout_seconds=30)

        first = new_id("preq")
        assert reserve(first) is not None and account.swept == []
        del store.settle
        second = new_id("preq")
        assert reserve(second) is not None and account.swept == [stuck]
        states = {e.entry_id: e.state for e in store.entries(task_id=task_id)}
        assert states == {stuck: SpendState.UNKNOWN, first: SpendState.RESERVED, second: SpendState.RESERVED}
    finally:
        dispose(sessions)


async def test_a_lesson_completes_while_every_sweep_fails(container, caplog) -> None:
    """End to end: every paid call of a lesson (and its diagnostic rounds) is reserved, sent and settled while the
    store's sweep fails on each reservation. Before the fix the first model call failed with the sweep's error."""
    calls = []

    def broken_sweep(now):
        calls.append(now)
        raise RuntimeError("storage hiccup during the sweep")

    container.spend.expire_abandoned = broken_sweep
    with caplog.at_level(logging.WARNING, logger="app.observability.spend"):
        task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED
    entries = container.spend.entries(task_id=task.task_id)
    assert entries and len(calls) >= len(entries)  # each reservation tried to sweep
    assert all(e.state == SpendState.SETTLED for e in entries)
    reserved, settled = totals(container.spend, f"task:{task.task_id}")
    assert reserved == 0 and settled > 0
    assert not [e for e in container.task_service.events(task.task_id) if e.type == "spend.abandoned"]
    assert any("sweep failed" in r.getMessage() for r in caplog.records)
