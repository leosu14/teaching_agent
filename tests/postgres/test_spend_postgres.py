"""The spend ledger on PostgreSQL (PR19), opt-in: skipped unless TA_TEST_POSTGRES_URL names a database the test may
write to (e.g. postgresql+psycopg://user@localhost/ta_test; needs a driver such as `pip install "psycopg[binary]"`).

The same races as tests/integration/test_spend_concurrency.py, where row locks (not SQLite's single writer)
serialize the conditional increments, plus the lease fence, the in-flight idempotency key (a partial unique index)
and the abandoned-reservation sweep. Every run uses fresh ids, so the database can be reused.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from datetime import timedelta

import pytest

from app.schemas.common import new_id, utcnow
from app.schemas.lease import LeaseLost
from app.schemas.spend import Charge, SpendInFlight, SpendReservation, SpendSettlement, SpendState
from app.schemas.task import Task
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlSpendRepository, SqlTaskRepository
from tests.integration.test_spend_concurrency import EACH, WORKERS, _process, check, race

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL spend tests")


@pytest.fixture
def store():
    sessions = create_db(URL)
    yield SqlSpendRepository(sessions), SqlTaskRepository(sessions)
    dispose(sessions)


def test_threads_never_reserve_past_the_limit_on_postgres() -> None:
    dispose(create_db(URL))
    scope = f"session:{new_id('pg')}"
    results: list[tuple[int, int]] = []
    barrier = threading.Barrier(WORKERS)

    def worker() -> None:
        results.append(race(URL, scope, EACH, settle_half=True, start=barrier))

    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert len(results) == WORKERS
    check(URL, scope, sum(a for a, _ in results))


def test_processes_never_reserve_past_the_limit_on_postgres() -> None:
    dispose(create_db(URL))
    scope = f"session:{new_id('pg')}"
    ctx = multiprocessing.get_context("spawn")
    out, start = ctx.Queue(), ctx.Barrier(4)
    procs = [ctx.Process(target=_process, args=(URL, scope, EACH, out, start)) for _ in range(4)]
    for p in procs:
        p.start()
    results = [out.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    check(URL, scope, sum(a for a, _ in results))


def _reservation(task_id: str, entry_id: str, *, key: str | None = None, fence=None, ttl: float = 60):
    charge = Charge(scope_key=f"task:{task_id}", metric="llm_tokens", amount=100, check=100, limit=10_000)
    return SpendReservation(entry_id=entry_id, operation_id=new_id("pop"), attempt=1, idempotency_key=key,
                            capability="llm", operation="generate", provider="p", task_id=task_id,
                            worker=fence.owner if fence else None, lease_fence=fence.fence if fence else None,
                            charges=[charge], estimate={"llm_tokens": 100},
                            reserved_until=utcnow() + timedelta(seconds=ttl))


def test_fence_idempotency_and_sweep_on_postgres(store) -> None:
    spend, tasks = store
    task_id = new_id("task")
    assert tasks.create(Task(task_id=task_id, user_id="u", learner_id="l", request="r"))
    lease = tasks.claim(task_id, "w1", timedelta(minutes=5))
    a, b = new_id("preq"), new_id("preq")
    key = f"video:{new_id('k')}"
    spend.reserve(_reservation(task_id, a, key=key, fence=lease), fence=lease, now=utcnow())
    spend.reserve(_reservation(task_id, a, key=key, fence=lease), fence=lease, now=utcnow())  # no-op
    with pytest.raises(SpendInFlight):
        spend.reserve(_reservation(task_id, b, key=key, fence=lease), fence=lease, now=utcnow())
    done = SpendSettlement(state=SpendState.SETTLED, charged={"llm_tokens": 40})
    assert spend.settle(a, done, now=utcnow()).outcome == "applied"
    assert spend.settle(a, done, now=utcnow()).outcome == "duplicate"

    later = utcnow() + timedelta(minutes=6)
    newer = SqlTaskRepository(tasks._sessions, lambda: later).claim(task_id, "w2", timedelta(minutes=5))
    assert newer is not None
    with pytest.raises(LeaseLost):
        spend.reserve(_reservation(task_id, b, fence=lease), fence=lease, now=later)

    c = new_id("preq")
    spend.reserve(_reservation(task_id, c, ttl=1), fence=None, now=utcnow())
    later = utcnow() + timedelta(seconds=5)
    while spend.expire_abandoned(later):  # a reused database may hold older expired entries: a sweep takes 100
        pass
    [swept] = [e for e in spend.entries(task_id=task_id) if e.entry_id == c]
    assert swept.state == SpendState.UNKNOWN and swept.error_type == "Abandoned"
    totals = spend.totals([f"task:{task_id}"])[f"task:{task_id}"]["llm_tokens"]
    assert (totals.reserved, totals.settled) == (0, 140)
