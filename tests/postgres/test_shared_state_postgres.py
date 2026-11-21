"""Shared state of several processes on PostgreSQL (issue #66 PR 2): the rate limiter, the research cache and the
progress-stream caps, with real separate processes (spawned, each with its own engine) on one database.

Opt-in like the other PostgreSQL tests (TA_TEST_POSTGRES_URL). Each test runs on a fresh database (`fresh_db`), so
the global stream cap and the rate-limit keys start empty.

- Rate limiter: the starts admitted per minute and the requests in flight for one (capability, provider), summed over
  every process, never exceed the limits (racing admissions; a live in-flight counter kept in the database by the
  requests themselves); a killed process's concurrency slot frees itself at its expiry in database time.
- Fail closed (D3): with the limiter's database unreachable, or its key's lock held past the decision timeout, zero
  provider calls are made and nothing is reserved; the failure is retryable.
- Research cache: an entry written by one process is read by another for the same learner only; never after 7 days of
  database time; the learner's deletion leaves none, and a write racing the deletion is refused.
- Stream caps: across two API processes the 5th stream of one principal and the 201st stream overall are refused;
  racing acquisitions for one principal admit exactly 4; a killed process's slots are reusable once their TTL ran out.
- Migration 0006 up and down.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from multiprocessing import get_context

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.schemas.common import new_id
from app.storage.db import create_db, dispose
from app.storage.migrate import alembic_config, upgrade

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL shared-state tests")
SHORT = timedelta(seconds=2)


def spawn(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))


def wait_until(start_at: float) -> None:
    """A start line for racing processes (wall clock: when to start, never a decision)."""
    while time.time() < start_at:
        time.sleep(0.001)


# --- what one process does (each runs in its own spawned process) ---------------------------------------------------


def admit_many(url: str, key: str, n: int, start_at: float, rpm: int | None, conc: int | None) -> int:
    from app.storage.shared_state import SqlRateLimits

    sessions = create_db(url)
    try:
        limits = SqlRateLimits(sessions)
        wait_until(start_at)
        return sum(limits.admit(key, requests_per_minute=rpm, max_concurrency=conc, slot_ttl=timedelta(minutes=5))
                   .admitted for _ in range(n))
    finally:
        dispose(sessions)


def run_limited_requests(url: str, key: str, n: int, conc: int, hold: float, start_at: float) -> int:
    """`n` concurrent requests through the shared limiter; each, while admitted, keeps the in-flight count and its peak
    in the database (`pr2_probe`), so the peak is the true sum over every process."""
    from app.providers.core.ratelimit import SharedRateLimiter
    from app.storage.shared_state import SqlRateLimits

    sessions = create_db(url)
    limiter = SharedRateLimiter(SqlRateLimits(sessions), key, max_concurrency=conc, slot_ttl=SHORT)

    def count(delta: int) -> None:
        with sessions.begin() as s:
            s.execute(text("UPDATE pr2_probe SET n = n + :d, peak = GREATEST(peak, n + :d) WHERE k = :k"),
                      {"d": delta, "k": key})

    async def one() -> None:
        async with limiter.slot():
            count(1)
            await asyncio.sleep(hold)
            count(-1)

    async def all_of_them() -> None:
        await asyncio.gather(*(one() for _ in range(n)))

    try:
        wait_until(start_at)
        asyncio.run(all_of_them())
        return n
    finally:
        dispose(sessions)


def hold_slot_and_die(url: str, key: str) -> None:
    """Admit a concurrency slot (TTL 2 s) and die without releasing it."""
    from app.storage.shared_state import SqlRateLimits

    sessions = create_db(url)
    admitted = SqlRateLimits(sessions).admit(key, requests_per_minute=None, max_concurrency=1, slot_ttl=SHORT)
    assert admitted.admitted
    os._exit(0)


def acquire_streams(url: str, principals: list[str], start_at: float, ttl_s: float = 60.0) -> list[bool]:
    from app.storage.shared_state import SqlStreamSlots

    sessions = create_db(url)
    try:
        slots = SqlStreamSlots(sessions)
        wait_until(start_at)
        return [slots.acquire(p, per_principal=4, total=200, ttl=timedelta(seconds=ttl_s)) is not None
                for p in principals]
    finally:
        dispose(sessions)


def hold_streams_and_die(url: str, principal: str, n: int) -> None:
    from app.storage.shared_state import SqlStreamSlots

    sessions = create_db(url)
    slots = SqlStreamSlots(sessions)
    assert all(slots.acquire(principal, per_principal=4, total=200, ttl=SHORT) for _ in range(n))
    os._exit(0)


def cache_put(url: str, learner_id: str, task_id: str, key: str, body: str) -> None:
    from app.storage.shared_state import SqlResearchCacheStore
    from app.tools.research.cache import CACHE_TTL

    sessions = create_db(url)
    try:
        SqlResearchCacheStore(sessions).put(learner_id, key, body, ttl=CACHE_TTL, task_id=task_id)
    finally:
        dispose(sessions)


def cache_get(url: str, learner_id: str, key: str) -> str | None:
    from app.storage.shared_state import SqlResearchCacheStore

    sessions = create_db(url)
    try:
        return SqlResearchCacheStore(sessions).get(learner_id, key)
    finally:
        dispose(sessions)


def die_in(fn, *args) -> None:
    process = get_context("spawn").Process(target=fn, args=args)
    process.start()
    process.join(60)
    assert process.exitcode == 0


@pytest.fixture
def db(fresh_db):
    sessions = create_db(fresh_db)
    yield fresh_db, sessions
    dispose(sessions)


# --- the rate limiter ------------------------------------------------------------------------------------------------


def test_racing_admissions_from_eight_processes_respect_the_window_and_the_concurrency_limit(db) -> None:
    url, _ = db
    with spawn(8) as pool:
        start = time.time() + 3
        rpm = [pool.submit(admit_many, url, "llm:rpm", 5, start, 7, None) for _ in range(8)]
        assert sum(f.result() for f in rpm) == 7  # 40 attempts, 7 per minute: exactly 7 admitted overall
        start = time.time() + 1
        conc = [pool.submit(admit_many, url, "llm:conc", 3, start, None, 5) for _ in range(8)]
        assert sum(f.result() for f in conc) == 5  # 24 attempts, none released: exactly 5 in flight


def test_requests_in_flight_across_two_processes_never_exceed_the_limit(db) -> None:
    url, sessions = db
    with sessions.begin() as s:
        s.execute(text("CREATE TABLE pr2_probe (k text PRIMARY KEY, n int NOT NULL, peak int NOT NULL)"))
        s.execute(text("INSERT INTO pr2_probe VALUES ('tts:vendor', 0, 0)"))
    with spawn(2) as pool:
        start = time.time() + 3
        done = [pool.submit(run_limited_requests, url, "tts:vendor", 6, 3, 0.15, start) for _ in range(2)]
        assert sum(f.result() for f in done) == 12
    with sessions() as s:
        n, peak = s.execute(text("SELECT n, peak FROM pr2_probe")).one()
        left = s.scalar(text("SELECT count(*) FROM rate_limit_slots"))
    assert n == 0 and 2 <= peak <= 3, peak  # 12 requests from two processes, at most 3 at once
    assert left == 0  # every slot released


def test_a_killed_processs_concurrency_slot_frees_itself_at_its_expiry_in_database_time(db) -> None:
    from app.storage.shared_state import SqlRateLimits

    url, sessions = db
    die_in(hold_slot_and_die, url, "llm:dead")
    with sessions() as s:
        expires = s.scalar(text("SELECT expires_ms FROM rate_limit_slots WHERE limit_key = 'llm:dead'"))
    limits = SqlRateLimits(sessions)
    refused_at, admitted_at = [], None
    deadline = time.monotonic() + 10
    while admitted_at is None and time.monotonic() < deadline:
        with sessions() as s:
            now = int(s.scalar(text("SELECT extract(epoch FROM clock_timestamp()) * 1000")))
        if limits.admit("llm:dead", requests_per_minute=None, max_concurrency=1, slot_ttl=SHORT).admitted:
            admitted_at = now
        else:
            refused_at.append(now)
            time.sleep(0.05)
    assert refused_at and admitted_at is not None
    assert max(refused_at) < expires + 50  # refused while the dead holder's slot was live
    assert admitted_at >= expires - 50  # taken over only once it expired (tolerance: the reading precedes admit)


async def _unavailable(store) -> tuple[list, list, object]:
    from app.observability.events import EventBus
    from app.observability.scope import ExecutionScope, UsageLedger, active_scope
    from app.providers.core.errors import RateLimiterUnavailable, is_retryable
    from app.providers.core.invoker import ProviderInvoker
    from app.providers.core.ratelimit import SharedRateLimiter
    from app.schemas.providers import Capability, ProviderPolicy, ProviderUsage
    from app.schemas.spend import Estimate

    sent: list = []

    class Vendor:
        name = "vendor"
        requires_network = True

    async def vendor_call():
        sent.append(1)
        return "ok"

    def usage(_r, request_id):
        return ProviderUsage(provider="vendor", capability=Capability.LLM, operation="generate", request_id=request_id)

    def factory(capability, provider, policy):
        return SharedRateLimiter(store, f"{capability.value}:{provider}", max_concurrency=policy.max_concurrency,
                                 requests_per_minute=policy.requests_per_minute, slot_ttl=SHORT)

    async def no_sleep(_s):
        return None

    policy = ProviderPolicy(max_attempts=2, backoff_seconds=0, max_concurrency=4, requests_per_minute=100)
    inv = ProviderInvoker(policies={c: policy for c in Capability}, limiter_factory=factory, sleep=no_sleep)
    ledger = UsageLedger()
    scope = ExecutionScope(events=EventBus(), usage=ledger, task_id="t1")
    with active_scope(scope), pytest.raises(RateLimiterUnavailable) as raised:
        await inv.call(Vendor(), Capability.LLM, "generate", vendor_call, usage=usage, estimate=Estimate(llm_tokens=9))
    assert is_retryable(raised.value)
    return sent, ledger.account.store.entries(), raised.value


def test_with_the_limiter_database_unreachable_no_paid_call_is_made_and_nothing_is_reserved() -> None:
    from sqlalchemy.engine import make_url

    from app.storage.shared_state import SqlRateLimits

    engine = create_engine(make_url(URL).set(port=1))  # nothing listens there: every connection is refused
    try:
        sent, reserved, _ = asyncio.run(_unavailable(SqlRateLimits(sessionmaker(bind=engine))))
    finally:
        engine.dispose()
    assert sent == [] and reserved == []


def test_a_decision_that_times_out_fails_closed(db) -> None:
    """The key's lock held by a stuck transaction past the decision timeout (5 s): the admission fails, nothing is
    sent or reserved, and the limiter is usable again once the lock is free."""
    from app.storage.shared_state import RATE_LIMIT_LOCK_SPACE, SqlRateLimits, _key_lock

    url, sessions = db
    engine = create_engine(url)
    held = threading.Event()
    free = threading.Event()

    def stuck() -> None:
        with engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(:a, :b)"),
                         {"a": RATE_LIMIT_LOCK_SPACE, "b": _key_lock("llm:vendor")})
            held.set()
            free.wait(30)

    blocker = threading.Thread(target=stuck)
    blocker.start()
    try:
        assert held.wait(10)
        started = time.monotonic()
        sent, reserved, _ = asyncio.run(_unavailable(SqlRateLimits(sessions)))
        assert sent == [] and reserved == []
        assert time.monotonic() - started >= 2 * 4.5  # two attempts, each refused at the timeout
    finally:
        free.set()
        blocker.join()
        engine.dispose()
    assert SqlRateLimits(sessions).admit("llm:vendor", requests_per_minute=100, max_concurrency=4,
                                         slot_ttl=SHORT).admitted


# --- the research cache ----------------------------------------------------------------------------------------------


def _task(sessions, learner_id: str) -> str:
    from app.schemas.common import utcnow
    from app.storage.orm import TaskRow

    task_id = new_id("t")
    with sessions.begin() as s:
        s.add(TaskRow(task_id=task_id, user_id="u", learner_id=learner_id, status="RUNNING", created_at=utcnow(),
                      updated_at=utcnow(), body="{}"))
    return task_id


def test_the_cache_is_shared_across_processes_per_learner_for_seven_days_of_database_time(db) -> None:
    from app.storage.repositories import SqlLearnerDataRepository
    from app.storage.shared_state import SqlResearchCacheStore
    from tests.postgres.dbclock import ShiftedDbClock

    url, sessions = db
    learner, other = new_id("l"), new_id("l")
    key = "a" * 64
    with spawn(2) as pool:
        pool.submit(cache_put, url, learner, _task(sessions, learner), key, '["mine"]').result()
        assert pool.submit(cache_get, url, learner, key).result() == '["mine"]'  # another process reads it
        assert pool.submit(cache_get, url, other, key).result() is None  # never another learner's (A2)
    later = SqlResearchCacheStore(sessions, db_clock=ShiftedDbClock(timedelta(days=7)))
    assert later.get(learner, key) is None  # 7 days of database time (D4)
    almost = SqlResearchCacheStore(sessions, db_clock=ShiftedDbClock(timedelta(days=7, seconds=-30)))
    assert almost.get(learner, key) == '["mine"]'
    assert later.purge_expired() == 1 and later.entries(learner) == 0
    # The learner's deletion leaves no entry (the TTL does not replace it).
    cache_put(url, learner, _task(sessions, learner), key, '["again"]')
    data = SqlLearnerDataRepository(sessions)
    counts, _ = data.erase(learner)
    assert counts["research_cache"] == 1 and not data.holds_data(learner)
    with sessions() as s:
        assert s.scalar(text("SELECT count(*) FROM research_cache WHERE learner_id = :l"), {"l": learner}) == 0


def test_a_cache_write_racing_the_learners_deletion_is_refused(db) -> None:
    """The write takes the learner's deletion lock shared and needs its lesson task: a deletion that holds the lock
    makes it wait, and once the deletion commits it is refused, so nothing is left behind."""
    from app.schemas.lease import LearnerDataGone
    from app.storage.repositories import SqlLearnerDataRepository, _deletion_lock_key
    from app.storage.shared_state import SqlResearchCacheStore
    from app.tools.research.cache import CACHE_TTL

    url, sessions = db
    learner = new_id("l")
    task_id = _task(sessions, learner)
    engine = create_engine(url)
    outcome: list = []
    with engine.connect() as conn:
        tx = conn.begin()
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                     {"k": _deletion_lock_key(learner)})

        def write() -> None:
            try:
                SqlResearchCacheStore(sessions).put(learner, "b" * 64, "[]", ttl=CACHE_TTL, task_id=task_id)
                outcome.append("stored")
            except LearnerDataGone:
                outcome.append("refused")

        writer = threading.Thread(target=write)
        writer.start()
        time.sleep(0.5)
        assert outcome == []  # waiting on the deletion
        conn.execute(text("DELETE FROM tasks WHERE task_id = :t"), {"t": task_id})
        tx.commit()
        writer.join(10)
    engine.dispose()
    assert outcome == ["refused"] and not SqlLearnerDataRepository(sessions).holds_data(learner)


# --- the stream caps -------------------------------------------------------------------------------------------------


def test_stream_caps_hold_across_two_api_processes(db) -> None:
    url, _ = db
    with spawn(2) as pool:
        start = time.time() + 3
        a = pool.submit(acquire_streams, url, ["alice"] * 3, start)
        b = pool.submit(acquire_streams, url, ["alice"] * 3, start)
        got = a.result() + b.result()
        assert got.count(True) == 4  # the 5th and 6th of one principal are refused, whichever process
        # Fill the deployment-wide cap from both processes: 196 more slots for 49 principals, then the 201st.
        principals = [f"p{i}" for i in range(49) for _ in range(4)]
        start = time.time() + 2
        halves = [pool.submit(acquire_streams, url, principals[i::2], start) for i in range(2)]
        assert all(all(h.result()) for h in halves)
        assert pool.submit(acquire_streams, url, ["newcomer"], time.time()).result() == [False]  # the 201st


def test_racing_streams_of_one_principal_from_eight_processes_admit_exactly_four(db) -> None:
    url, _ = db
    with spawn(8) as pool:
        start = time.time() + 3
        got = [r for f in [pool.submit(acquire_streams, url, ["bob"], start) for _ in range(8)] for r in f.result()]
    assert got.count(True) == 4


def test_a_killed_processs_stream_slots_are_reusable_once_their_ttl_ran_out(db) -> None:
    from app.storage.shared_state import SqlStreamSlots

    url, sessions = db
    die_in(hold_streams_and_die, url, "carol", 4)
    slots = SqlStreamSlots(sessions)
    assert slots.acquire("carol", per_principal=4, total=200, ttl=SHORT) is None
    deadline = time.monotonic() + 10
    reused = None
    while reused is None and time.monotonic() < deadline:
        reused = slots.acquire("carol", per_principal=4, total=200, ttl=SHORT)
        time.sleep(0.05)
    assert reused is not None and slots.open_count("carol") == 1  # the dead process's slots were deleted


# --- migration 0006 ----------------------------------------------------------------------------------------------


def test_migration_0006_up_and_down_on_postgresql(fresh_db) -> None:
    from alembic import command

    upgrade(fresh_db)
    tables = {"rate_limit_starts", "rate_limit_slots", "stream_slots", "research_cache"}

    def present() -> set[str]:
        engine = create_engine(fresh_db)
        try:
            return set(inspect(engine).get_table_names())
        finally:
            engine.dispose()

    assert tables <= present()
    command.downgrade(alembic_config(fresh_db), "0005_teaching_turn_lease")
    assert not tables & present()
    command.upgrade(alembic_config(fresh_db), "head")
    assert tables <= present()
