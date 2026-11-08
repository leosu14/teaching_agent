"""MVP item 3 (spend backlog F7) on PostgreSQL, opt-in: skipped unless TA_TEST_POSTGRES_URL names a database the test
may write to (see tests/postgres/test_spend_postgres.py).

A real database error inside the sweep (an over-long value rejected by PostgreSQL in the middle of a settlement
transaction) must roll back only that entry's settlement: the other abandoned entries are swept, the connection goes
back to the pool usable, and concurrent reservations that each hit the failing entry in their sweep still reserve,
with counters that equal the sum of their entries. Every run uses fresh ids, so the database can be reused.
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta

import pytest

from app.observability.spend import abandoned, task_account
from app.schemas.common import new_id, utcnow
from app.schemas.providers import Capability
from app.schemas.spend import Charge, Estimate, SpendReservation, SpendState
from app.schemas.usage import BudgetDefaults
from app.storage import repositories
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlSpendRepository

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL spend tests")


@pytest.fixture
def spend():
    sessions = create_db(URL)
    yield SqlSpendRepository(sessions)
    dispose(sessions)


def stuck(spend: SqlSpendRepository, task_id: str, n: int) -> list[str]:
    """`n` reservations of task `task_id` that nobody settles (they expire a second later)."""
    ids = [new_id("preq") for _ in range(n)]
    for entry_id in ids:
        charge = Charge(scope_key=f"task:{task_id}", metric="llm_tokens", amount=100, check=100, limit=None)
        spend.reserve(SpendReservation(
            entry_id=entry_id, operation_id=new_id("pop"), attempt=1, capability="llm", operation="generate",
            provider="p", task_id=task_id, charges=[charge], estimate={"llm_tokens": 100},
            reserved_until=utcnow() + timedelta(seconds=1)), fence=None, now=utcnow())
    return ids


@pytest.fixture
def poisoned(monkeypatch):
    """Settling the entries in the returned set as abandoned writes a value PostgreSQL rejects (error_type is a
    VARCHAR(64)): a real DataError in the middle of the settlement transaction."""
    bad: set[str] = set()

    def settlement(entry):
        s = abandoned(entry)
        return s.model_copy(update={"error_type": "X" * 200}) if entry.entry_id in bad else s

    monkeypatch.setattr(repositories, "abandoned", settlement)
    return bad


def sweep_all(spend: SqlSpendRepository, now) -> list[str]:
    """Sweep until a pass settles nothing (a reused database may hold older expired entries: a pass takes 100)."""
    out: list[str] = []
    while swept := spend.expire_abandoned(now):
        out += swept
    return out


def llm_tokens(spend: SqlSpendRepository, task_id: str) -> tuple[int, int]:
    t = spend.totals([f"task:{task_id}"])[f"task:{task_id}"]["llm_tokens"]
    return t.reserved, t.settled


def test_a_database_error_on_one_entry_leaves_the_rest_of_the_sweep_on_postgres(spend, poisoned) -> None:
    task_id = new_id("task")
    a, b, c = stuck(spend, task_id, 3)
    poisoned.add(b)
    later = utcnow() + timedelta(seconds=5)
    swept = sweep_all(spend, later)
    assert a in swept and c in swept and b not in swept
    states = {e.entry_id: e.state for e in spend.entries(task_id=task_id)}
    assert states == {a: SpendState.UNKNOWN, b: SpendState.RESERVED, c: SpendState.UNKNOWN}
    assert llm_tokens(spend, task_id) == (100, 200)  # the failed settlement rolled back entirely

    d = stuck(spend, task_id, 1)[0]  # the pooled connections are still usable after the error
    assert llm_tokens(spend, task_id) == (200, 200)

    poisoned.clear()
    assert set(sweep_all(spend, utcnow() + timedelta(seconds=5))) >= {b, d}
    assert llm_tokens(spend, task_id) == (0, 400)


def test_concurrent_reservations_go_ahead_while_their_sweeps_fail_on_postgres(spend, poisoned) -> None:
    """8 workers reserve at once; each one's inline sweep hits the poisoned abandoned entry and fails on it. Every
    reservation still commits, exactly once, and the counter equals the sum of the entries."""
    task_id = new_id("task")
    [bad] = stuck(spend, task_id, 1)
    poisoned.add(bad)
    later = utcnow() + timedelta(seconds=5)
    sweep_all(spend, later)  # clear any older expired entries of a reused database
    workers, each = 8, 5
    barrier = threading.Barrier(workers)
    reserved: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        account = task_account(spend, BudgetDefaults(), task_id=task_id, learner_id=None, budget=None, fence=None,
                               clock=lambda: later)
        barrier.wait()
        try:
            for _ in range(each):
                entry_id = new_id("preq")
                assert account.reserve(entry_id=entry_id, operation_id=new_id("pop"), attempt=1,
                                       capability=Capability.LLM, operation="generate", provider="p", model="m",
                                       estimate=Estimate(llm_tokens=100), timeout_seconds=30) is not None
                assert bad not in account.swept
                with lock:
                    reserved.append(entry_id)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert errors == [] and len(reserved) == workers * each
    entries = {e.entry_id: e for e in spend.entries(task_id=task_id)}
    assert set(entries) == {bad, *reserved}
    assert all(e.state == SpendState.RESERVED for e in entries.values())
    tokens = sum(e.estimate["llm_tokens"] for e in entries.values())
    assert llm_tokens(spend, task_id) == (tokens, 0) == ((workers * each + 1) * 100, 0)

    poisoned.clear()  # the storage recovers: the next sweep charges the abandoned entry once
    assert bad in sweep_all(spend, later)
    assert spend.entries(task_id=task_id) and llm_tokens(spend, task_id) == (workers * each * 100, 100)
