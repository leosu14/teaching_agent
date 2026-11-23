"""Database time for every lease, fence and spend-expiry decision (issue #66 PR 1, acceptance criterion 1).

- A static check: no function that decides on a lease, a fence or a reservation's expiry reads a process clock
  (`self._clock()`, `utcnow()`, `datetime.now()`, `time.time()`). Those decisions read `DbClock` (the database's
  clock on PostgreSQL) instead. A new lease predicate written against a process clock fails here.
- `DbClock` itself: the injected clock on SQLite, so the unit and integration tests that move time keep doing so.
- Migration 0005 up and down on SQLite (the PostgreSQL one is in tests/postgres/test_multi_replica_postgres.py).
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect

from app.storage.db import create_db, dispose
from app.storage.dbclock import DbClock
from app.storage.migrate import upgrade

ROOT = Path(__file__).resolve().parents[2]
# The modules whose lease, fence and expiry decisions must use database time. The in-memory stores of unit tests
# (app/teaching/repository.py, app/curriculum/repository.py, InMemorySpendStore) have no database: their injected
# clock is their only clock.
MODULES = ["app/storage/repositories.py", "app/services/learning_cycles.py", "app/services/sweeper.py",
           "app/services/teaching.py", "app/observability/spend.py",
           "app/storage/shared_state.py"]  # issue #66 PR 2: rate-limit, stream-slot and research-cache expiry
# What a lease, fence or expiry decision names.
LEASE_NAMES = {"lease_expires_ms", "turn_lease_expires_ms", "reserved_until_ms", "reserved_until", "available_at_ms",
               "_require_fence", "_fence_current", "lease_live", "until", "claim_turns", "renew_turns", "turn_lease",
               "expires_ms", "started_ms"}
# Functions that name a lease column but read the process clock for something that is not a lease decision.
ALLOWED = {
    # The job's requested backoff (`job.available_at`, from the caller's clock) is turned into a delay from the
    # caller's clock, then added to database time: the decision is database time plus that delay.
    ("app/storage/repositories.py", "SqlTaskJobRepository.enqueue"),
    # The in-memory store has no database: its injected clock is its only clock.
    ("app/observability/spend.py", "InMemorySpendStore.expire_abandoned"),
    # Without a store clock the account's clock is the only one (`database_now()` is tried first).
    ("app/observability/spend.py", "SpendAccount.now"),
    # Stamps the turns it produces with the service clock (recorded, not decided); the lease decisions are the
    # repository's (`claim_turns`, the fenced `apply`), in database time.
    ("app/services/teaching.py", "TeachingSessionService._turns"),
}


def _process_clock_reads(node: ast.AST) -> list[str]:
    found = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr == "_clock" and isinstance(f.value, ast.Name) and f.value.id == "self":
            found.append("self._clock()")
        elif isinstance(f, ast.Name) and f.id == "utcnow":
            found.append("utcnow()")
        elif isinstance(f, ast.Attribute) and f.attr == "now" and isinstance(f.value, ast.Name) \
                and f.value.id == "datetime":
            found.append("datetime.now()")
        elif isinstance(f, ast.Attribute) and f.attr == "time" and isinstance(f.value, ast.Name) and f.value.id == "time":
            found.append("time.time()")
    return found


def _names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
    return out


def _functions(tree: ast.Module):
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield top.name, top
        elif isinstance(top, ast.ClassDef):
            for item in top.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{top.name}.{item.name}", item


def test_no_lease_fence_or_expiry_decision_reads_a_process_clock() -> None:
    offenders = []
    for module in MODULES:
        tree = ast.parse((ROOT / module).read_text())
        for name, fn in _functions(tree):
            if (module, name) in ALLOWED or not (_names(fn) & LEASE_NAMES):
                continue
            reads = _process_clock_reads(fn)
            if reads:
                offenders.append(f"{module}:{fn.lineno} {name} reads {', '.join(sorted(set(reads)))}")
    assert offenders == [], "lease, fence and expiry decisions must use DbClock:\n" + "\n".join(offenders)


def test_the_check_finds_a_process_clock_in_a_lease_predicate() -> None:
    """The static check itself: a predicate against the process clock is found."""
    fn = ast.parse("def claim(self):\n    return TaskRow.lease_expires_ms <= _ms(self._clock())\n").body[0]
    assert _names(fn) & LEASE_NAMES and _process_clock_reads(fn) == ["self._clock()"]


def test_on_sqlite_database_time_is_the_injected_clock(tmp_path) -> None:
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    try:
        at = datetime(2026, 11, 20, 23, 59, 59, tzinfo=timezone.utc)
        clock = DbClock(lambda: at)
        with sessions() as s:
            assert clock.now(s) == at and clock.database_now(s) is None
        assert clock.read(sessions) is None
    finally:
        dispose(sessions)


def test_migration_0005_adds_and_removes_the_teaching_turn_lease(tmp_path) -> None:
    from alembic import command

    from app.storage.migrate import alembic_config as _config

    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    upgrade(url)

    def columns() -> set[str]:
        engine = create_engine(url)
        try:
            return {c["name"] for c in inspect(engine).get_columns("teaching_sessions")}
        finally:
            engine.dispose()

    assert {"turn_lease_token", "turn_lease_expires_ms"} <= columns()
    command.downgrade(_config(url), "0004_class_assignments")
    assert not {"turn_lease_token", "turn_lease_expires_ms"} & columns()
    command.upgrade(_config(url), "head")
    assert {"turn_lease_token", "turn_lease_expires_ms"} <= columns()


def test_a_lease_ends_a_ttl_after_database_time(tmp_path) -> None:
    """A claim's end is database time plus the TTL (here SQLite: the injected clock)."""
    from app.schemas.task import Task
    from app.storage.repositories import SqlTaskRepository

    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    try:
        at = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)
        tasks = SqlTaskRepository(sessions, lambda: at)
        tasks.create(Task(task_id="t1", user_id="u", learner_id="l", request="r"))
        lease = tasks.claim("t1", "w", timedelta(minutes=5))
        assert lease is not None and lease.until == at + timedelta(minutes=5)
        assert tasks.lease("t1") == lease
    finally:
        dispose(sessions)
