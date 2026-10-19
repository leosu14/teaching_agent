"""Concurrent reservations against one budget on one SQLite database (PR19): threads and separate processes, each
with its own engine, race for the same counters. The conditional increment admits exactly as many reservations as
fit, never more, and the counters equal the sum of the entries afterwards.

These guarantees are tested here with SQLite only (its single writer serializes the transactions). The same test on
PostgreSQL, where row locks do it, runs from tests/postgres when TA_TEST_POSTGRES_URL is set.
"""

from __future__ import annotations

import multiprocessing
import threading
from datetime import timedelta

from app.schemas.common import new_id, utcnow
from app.schemas.spend import Charge, ReservationRefused, SpendReservation, SpendSettlement, SpendState
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlSpendRepository

LIMIT = 1000
AMOUNT = 70  # 14 reservations fit (980), the 15th would not
WORKERS = 8
EACH = 6  # 48 attempts in all


def reservation(scope: str) -> SpendReservation:
    entry_id = new_id("preq")
    # One counter only: a second one locked first would serialize the transactions by itself and hide a
    # check-then-write race in the first.
    charges = [Charge(scope_key=scope, metric="llm_tokens", amount=AMOUNT, check=AMOUNT, limit=LIMIT)]
    return SpendReservation(entry_id=entry_id, operation_id=new_id("pop"), attempt=1, capability="llm",
                            operation="generate", provider="p", session_id="s1", charges=charges,
                            estimate={"llm_tokens": AMOUNT},
                            reserved_until=utcnow() + timedelta(minutes=5))


def race(url: str, scope: str, n: int, settle_half: bool = False, start=None) -> tuple[int, int]:
    """One worker: its own engine, `n` reservation attempts once every worker is ready (`start`, a barrier).
    Returns (admitted, refused)."""
    sessions = create_db(url)
    store = SqlSpendRepository(sessions)
    store.totals([scope])  # connected before the start, so the workers really overlap
    if start is not None:
        start.wait()
    admitted = refused = 0
    try:
        for i in range(n):
            r = reservation(scope)
            try:
                store.reserve(r, fence=None, now=utcnow())
            except ReservationRefused:
                refused += 1
                continue
            admitted += 1
            if settle_half and i % 2 == 0:  # settled at the estimate: the room stays taken
                store.settle(r.entry_id, SpendSettlement(state=SpendState.SETTLED, charged=dict(r.estimate)),
                             now=utcnow())
    finally:
        dispose(sessions)
    return admitted, refused


def _process(url: str, scope: str, n: int, out, start) -> None:
    out.put(race(url, scope, n, settle_half=True, start=start))


def check(url: str, scope: str, admitted: int) -> None:
    sessions = create_db(url)
    store = SqlSpendRepository(sessions)
    try:
        assert admitted == LIMIT // AMOUNT
        entries = store.entries(session_id="s1")
        mine = [e for e in entries if scope in e.scope_keys]
        assert len(mine) == admitted
        t = store.totals([scope])[scope]["llm_tokens"]
        assert t.reserved + t.settled == admitted * AMOUNT <= LIMIT
        assert t.settled == sum(e.charged.get("llm_tokens", 0) for e in mine if e.state != SpendState.RESERVED)
    finally:
        dispose(sessions)


def test_threads_never_reserve_past_the_limit(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    dispose(create_db(url))
    results: list[tuple[int, int]] = []
    barrier = threading.Barrier(WORKERS)

    def worker() -> None:
        results.append(race(url, "session:threads", EACH, settle_half=True, start=barrier))

    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert len(results) == WORKERS
    assert sum(r for _, r in results) == WORKERS * EACH - LIMIT // AMOUNT
    check(url, "session:threads", sum(a for a, _ in results))


def test_processes_never_reserve_past_the_limit(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    dispose(create_db(url))
    ctx = multiprocessing.get_context("spawn")
    out, start = ctx.Queue(), ctx.Barrier(4)
    procs = [ctx.Process(target=_process, args=(url, "session:procs", EACH, out, start)) for _ in range(4)]
    for p in procs:
        p.start()
    results = [out.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    check(url, "session:procs", sum(a for a, _ in results))
