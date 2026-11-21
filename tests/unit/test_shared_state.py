"""Shared state of several processes (issue #66 PR 2) on SQLite, with the injected clock as database time.

- The shared rate limiter: the requests-per-minute window and the concurrency slots counted in the database (two
  limiters on one database, as two processes would be), a killed holder's slot freed at its expiry, renewals.
- Fail closed (D3): a store that cannot decide makes the call fail retryably before anything is reserved or sent, with
  a fixed log message; zero provider calls.
- The research cache on the database: per learner, 7 days in database time, erased with the learner, and refused
  for a lesson task that was deleted.
- Shared stream slots: the caps across two services on one database, a killed process's slot reusable after its TTL,
  release on end, disconnect and reconnect, a lost slot stops the stream, an unavailable store refuses with 503, and a
  stream writes nothing but its slot row (decision A3).
- The startup check: no process-local implementation where several processes share PostgreSQL.

The same guarantees with real separate processes on PostgreSQL are in tests/postgres/test_shared_state_postgres.py.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event as orm_event
from sqlalchemy import select

from app.config.routing import ConfigError
from app.config.settings import Settings
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.providers.core.errors import RateLimiterUnavailable, is_retryable
from app.providers.core.invoker import ProviderInvoker
from app.providers.core.ratelimit import UNAVAILABLE, SharedRateLimiter
from app.schemas.events import Event, EventType
from app.schemas.lease import LearnerDataGone
from app.schemas.providers import Capability, ProviderPolicy, ProviderUsage
from app.schemas.research import SearchQuery
from app.schemas.spend import Estimate
from app.services.progress_stream import (
    SharedStreamSlots,
    StreamSlotsUnavailable,
    TooManyStreams,
)
from app.storage.db import create_db, dispose
from app.storage.orm import ResearchCacheRow, StreamSlotRow, TaskRow
from app.storage.repositories import SqlLearnerDataRepository
from app.storage.shared_state import (
    SharedStateSweeper,
    SqlRateLimits,
    SqlResearchCacheStore,
    SqlStreamSlots,
    principal_hash,
)
from app.tools.research.cache import CACHE_TTL, SharedResearchCache, research_cache_key
from tests.unit.test_progress_stream import Clock, FakeEvents, FakeReader, Run, service
from tests.unit.test_research import result

T0 = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
TTL = timedelta(seconds=30)


class Time:
    """The injected clock: database time on SQLite."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def db(tmp_path):
    factory = create_db(f"sqlite:///{tmp_path / 'shared.db'}")
    yield factory
    dispose(factory)


@pytest.fixture
def time_():
    return Time()


# --- the rate limiter's store ----------------------------------------------------------------------------------------


def test_the_window_is_counted_across_limiters_on_one_database(db, time_) -> None:
    a, b = SqlRateLimits(db, time_), SqlRateLimits(db, time_)  # two processes
    admitted = [x.admit("llm:p", requests_per_minute=3, max_concurrency=None, slot_ttl=TTL) for x in (a, b, a, b)]
    assert [d.admitted for d in admitted] == [True, True, True, False]
    assert admitted[3].reason == "requests_per_minute" and admitted[3].wait_seconds == 60.0
    time_.advance(seconds=59)
    late = b.admit("llm:p", requests_per_minute=3, max_concurrency=None, slot_ttl=TTL)
    assert not late.admitted and late.wait_seconds == 1.0
    assert a.admit("llm:other", requests_per_minute=3, max_concurrency=None, slot_ttl=TTL).admitted  # per key
    time_.advance(seconds=1)  # the first starts leave the window at 60 s of database time
    assert b.admit("llm:p", requests_per_minute=3, max_concurrency=None, slot_ttl=TTL).admitted


def test_concurrency_slots_are_leases_freed_on_release_or_at_expiry(db, time_) -> None:
    a, b = SqlRateLimits(db, time_), SqlRateLimits(db, time_)

    def admit(x):
        return x.admit("tts:p", requests_per_minute=None, max_concurrency=2, slot_ttl=TTL)

    first, second, third = admit(a), admit(b), admit(a)
    assert first.admitted and second.admitted and not third.admitted and third.reason == "concurrency"
    a.release(first.slot_id)
    fourth = admit(b)
    assert fourth.admitted and a.in_flight("tts:p") == 2
    # `second`'s process dies: it never releases. Its renewals stop, and the slot frees itself at its expiry.
    time_.advance(seconds=20)
    assert b.renew(fourth.slot_id, TTL)  # the live holder renews
    time_.advance(seconds=10)
    assert a.in_flight("tts:p") == 1
    assert admit(a).admitted
    assert not admit(b).admitted  # the renewed slot still counts
    assert not a.renew(second.slot_id, TTL)  # the dead one is gone: a late renew cannot revive it


def test_the_sweeper_deletes_expired_shared_state(db, time_) -> None:
    limits, slots, cache = SqlRateLimits(db, time_), SqlStreamSlots(db, time_), SqlResearchCacheStore(db, time_)
    limits.admit("k", requests_per_minute=5, max_concurrency=5, slot_ttl=TTL)
    slots.acquire("p", per_principal=4, total=10, ttl=TTL)
    add_task(db, "t1", "l1")
    cache.put("l1", "k" * 64, "[]", ttl=CACHE_TTL, task_id="t1")
    sweeper = SharedStateSweeper(limits, slots, cache)
    assert sweeper.sweep() == type(sweeper.sweep())()  # nothing expired yet
    time_.advance(days=7)
    done = sweeper.sweep()
    assert (done.rate_limit_rows, done.stream_slots, done.research_cache_entries) == (2, 1, 1)


# --- the shared limiter and fail closed ------------------------------------------------------------------------------


class Vendor:
    name = "vendor"
    requires_network = True


def usage(_result, request_id: str) -> ProviderUsage:
    return ProviderUsage(provider="vendor", capability=Capability.LLM, operation="generate", model="m",
                         request_id=request_id, input_tokens=1, output_tokens=1)


async def no_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


def invoker(factory, *, attempts: int = 1, max_concurrency: int | None = None, rpm: int | None = None):
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    policy = ProviderPolicy(max_attempts=attempts, backoff_seconds=0, max_concurrency=max_concurrency,
                            requests_per_minute=rpm)
    inv = ProviderInvoker(policies={c: policy for c in Capability}, sleep=no_sleep, limiter_factory=factory)
    ledger = UsageLedger()
    return inv, ExecutionScope(events=bus, usage=ledger, task_id="t1"), ledger, seen


def shared(store, **kw):
    def build(capability, provider, policy):
        return SharedRateLimiter(store, f"{capability.value}:{provider}", requests_per_minute=policy.requests_per_minute,
                                 max_concurrency=policy.max_concurrency, slot_ttl=TTL, sleep=no_sleep, **kw)
    return build


class Down:
    """A store whose database fails (or times out) on every decision."""

    def __init__(self) -> None:
        self.calls = 0

    def admit(self, *_a, **_k):
        self.calls += 1
        raise OSError("could not connect to server: postgresql://user:secret@db/x")

    def renew(self, *_a, **_k):
        raise OSError("down")

    def release(self, *_a, **_k):
        raise OSError("down")


async def test_an_unavailable_limiter_store_fails_closed_before_any_spend_or_call(caplog) -> None:
    store = Down()
    sent = []

    async def vendor_call():
        sent.append(1)
        return "ok"

    inv, scope, ledger, seen = invoker(shared(store), attempts=3, max_concurrency=2)
    caplog.set_level(logging.WARNING)
    with active_scope(scope), pytest.raises(RateLimiterUnavailable) as raised:
        await inv.call(Vendor(), Capability.LLM, "generate", vendor_call, usage=usage, model="m",
                       estimate=Estimate(llm_tokens=500))
    assert sent == []  # zero paid provider calls
    assert ledger.account.store.entries() == []  # nothing reserved, so nothing to settle
    assert store.calls == 3 and is_retryable(raised.value) and raised.value.possibly_billed is False
    failed = [e for e in seen if e.type == EventType.PROVIDER_REQUEST_FAILED]
    assert [e.data["will_retry"] for e in failed] == [True, True, False]
    assert all(e.data["reason"] == "rate_limiter_unavailable" for e in failed)
    assert not [e for e in seen if e.type == EventType.PROVIDER_REQUEST_STARTED]
    messages = [r.getMessage() for r in caplog.records if UNAVAILABLE in r.getMessage()]
    assert len(messages) == 3 and all("secret" not in m for m in messages)  # fixed message: no driver text


async def test_an_unlimited_capability_needs_no_decision() -> None:
    store = Down()
    inv, scope, _, _ = invoker(shared(store))

    async def ok():
        return "ok"

    with active_scope(scope):
        assert await inv.call(Vendor(), Capability.LLM, "generate", ok, usage=usage, model="m") == "ok"
    assert store.calls == 0


async def test_two_limiters_on_one_database_never_exceed_the_concurrency_limit(db, time_) -> None:
    running, peak = [0], [0]

    async def vendor_call():
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.005)
        running[0] -= 1
        return "ok"

    calls = []
    for _ in range(2):  # two processes: each its own invoker and limiters, one database
        inv, scope, ledger, _ = invoker(shared(SqlRateLimits(db, time_)), max_concurrency=2)
        for _ in range(4):
            async def one(inv=inv, scope=scope):
                with active_scope(scope):
                    return await inv.call(Vendor(), Capability.LLM, "generate", vendor_call, usage=usage, model="m")
            calls.append(one())
    assert await asyncio.gather(*calls) == ["ok"] * 8
    assert peak[0] == 2
    assert SqlRateLimits(db, time_).in_flight("llm:vendor") == 0  # every slot released


async def test_a_full_window_waits_for_the_oldest_start_to_leave(db, time_) -> None:
    waits = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        time_.advance(seconds=seconds)

    limiter = SharedRateLimiter(SqlRateLimits(db, time_), "llm:p", requests_per_minute=2, slot_ttl=TTL, sleep=sleep)
    reported = []
    for _ in range(3):
        async with limiter.slot(lambda reason, s: reported.append((reason, s))):
            time_.advance(seconds=1)
    assert waits == [58.0] and reported == [("requests_per_minute", 58.0)]


async def test_a_held_slot_is_renewed_while_its_request_runs(db, time_) -> None:
    store = SqlRateLimits(db, time_)
    renewed = []
    original = store.renew

    def renew(slot_id, ttl):
        renewed.append(slot_id)
        time_.advance(seconds=25)  # each beat, database time moves almost a TTL
        return original(slot_id, ttl)

    store.renew = renew  # type: ignore[method-assign]
    limiter = SharedRateLimiter(store, "llm:p", max_concurrency=1, slot_ttl=timedelta(seconds=0.03))
    other = SqlRateLimits(db, time_)
    async with limiter.slot():
        await asyncio.sleep(0.1)
        assert len(renewed) >= 2
        # 50+ s of database time later, the slot is still held: another process is not admitted.
        assert not other.admit("llm:p", requests_per_minute=None, max_concurrency=1, slot_ttl=TTL).admitted
    assert other.admit("llm:p", requests_per_minute=None, max_concurrency=1, slot_ttl=TTL).admitted


# --- the research cache ----------------------------------------------------------------------------------------------


def add_task(sessions, task_id: str, learner_id: str) -> None:
    with sessions.begin() as s:
        s.add(TaskRow(task_id=task_id, user_id="u", learner_id=learner_id, status="RUNNING", created_at=T0,
                      updated_at=T0, body="{}"))


def test_the_shared_cache_is_per_learner_expires_in_database_time_and_is_erased_with_the_learner(db, time_) -> None:
    store = SqlResearchCacheStore(db, time_)
    a, b = SharedResearchCache(store), SharedResearchCache(SqlResearchCacheStore(db, time_))  # two processes
    add_task(db, "t1", "l1")
    add_task(db, "t2", "l2")
    k = research_cache_key(SearchQuery(text="Football"), providers=["mock"])
    stored = [result("https://a.example/x")]
    a.set("l1", k, stored, task_id="t1")
    assert b.get("l1", k) == stored  # shared across processes
    assert b.get("l2", k) is None  # never another learner's (A2)
    time_.advance(days=7, seconds=-1)
    assert b.get("l1", k) == stored
    time_.advance(seconds=1)
    assert b.get("l1", k) is None  # 7 days (D4)
    a.set("l1", k, stored, task_id="t1")  # an expired entry is replaced
    b.set("l2", k, stored, task_id="t2")
    assert b.get("l1", k) == stored
    deletion = SqlLearnerDataRepository(db, time_)
    assert deletion.holds_data("l1")
    counts, _ = deletion.erase("l1")
    assert counts["research_cache"] == 1 and not deletion.holds_data("l1")
    assert store.entries("l1") == 0 and b.get("l2", k) == stored  # the other learner's entry stays


def test_a_cache_write_for_a_deleted_lesson_task_is_refused(db, time_) -> None:
    store = SqlResearchCacheStore(db, time_)
    with pytest.raises(LearnerDataGone):
        store.put("l1", "k" * 64, "[]", ttl=CACHE_TTL, task_id="gone")
    SharedResearchCache(store).set("l1", "k" * 64, [], task_id="gone")  # the cache swallows it: not cached
    with db() as s:
        assert s.scalars(select(ResearchCacheRow)).all() == []


# --- shared stream slots ---------------------------------------------------------------------------------------------


def slots(db, time_, **kw) -> SharedStreamSlots:
    return SharedStreamSlots(SqlStreamSlots(db, time_), per_principal=kw.get("per_principal", 4),
                             total=kw.get("total", 200), ttl=kw.get("ttl", TTL))


def test_stream_caps_hold_across_api_processes(db, time_) -> None:
    one, two = slots(db, time_, total=6), slots(db, time_, total=6)
    held = [one.acquire("alice"), two.acquire("alice"), one.acquire("alice"), two.acquire("alice")]
    with pytest.raises(TooManyStreams):
        one.acquire("alice")  # the 5th of one principal, whichever process
    held += [one.acquire("bob"), two.acquire("carol")]
    with pytest.raises(TooManyStreams):
        two.acquire("dave")  # the total
    assert one.open_count() == 6 and two.open_count("alice") == 4
    held[0].release()
    held[0].release()  # idempotent
    two.acquire("dave")
    with db() as s:  # a hash of the principal, never the identity
        assert {r.principal_hash for r in s.scalars(select(StreamSlotRow))} == {
            principal_hash(p) for p in ("alice", "bob", "carol", "dave")}


def test_a_killed_processs_slots_are_reusable_after_their_ttl(db, time_) -> None:
    dead = slots(db, time_, per_principal=2)
    dead.acquire("alice"), dead.acquire("alice")  # its process dies: never released, never renewed
    live = slots(db, time_, per_principal=2)
    with pytest.raises(TooManyStreams):
        live.acquire("alice")
    time_.advance(seconds=29)
    with pytest.raises(TooManyStreams):
        live.acquire("alice")
    time_.advance(seconds=1)
    live.acquire("alice")


class DownSlots:
    def acquire(self, *_a, **_k):
        raise OSError("down")

    def open_count(self, principal=None):
        raise OSError("down")


def test_an_unavailable_slot_store_refuses_the_stream() -> None:
    svc = SharedStreamSlots(DownSlots(), per_principal=4, total=200, ttl=TTL)  # type: ignore[arg-type]
    with pytest.raises(StreamSlotsUnavailable):
        svc.acquire("alice")


def shared_service(db, time_, clock: Clock, **limits):
    svc = service(FakeEvents(clock), clock, **limits)
    svc.slots = slots(db, time_, per_principal=limits.get("max_per_principal", 4),
                      ttl=timedelta(seconds=limits.get("slot_ttl_seconds", 60.0)))
    return svc


def test_a_shared_slot_is_released_at_the_end_on_disconnect_and_at_reconnect(db, time_) -> None:
    async def go() -> None:
        clock = Clock()
        svc = shared_service(db, time_, clock, max_seconds=5)
        ended = FakeReader({"task:t1": "t1"})
        ended.stopped = True
        r = Run(svc, ended)
        await r.until(lambda f: f.get("event") == "end")
        assert await r.next() is None and svc.slots.open_count() == 0  # normal close
        r = Run(svc, FakeReader({"task:t1": "t1"}))
        await r.next()
        assert svc.slots.open_count() == 1
        await r.gen.aclose()  # the client disconnected
        assert svc.slots.open_count() == 0
        r = Run(svc, FakeReader({"task:t1": "t1"}))
        await r.until(lambda f: f.get("event") == "reconnect")  # the maximum lifetime
        assert await r.next() is None and svc.slots.open_count() == 0
        failing = FakeReader({"task:t1": "t1"})
        r = Run(svc, failing)
        await r.next()
        failing.fail = RuntimeError("poll failed")
        assert await r.until(lambda f: False) is None and svc.slots.open_count() == 0  # a failed poll

    asyncio.run(go())


def test_a_stream_renews_its_slot_and_stops_when_the_slot_is_lost(db, time_) -> None:
    async def go() -> None:
        clock = Clock()
        svc = shared_service(db, time_, clock, max_seconds=600, slot_ttl_seconds=3.0)
        r = Run(svc, FakeReader({"task:t1": "t1"}))
        await r.next()
        for _ in range(5):  # 1 s polls, a renew every second: past the TTL in database time, still held
            time_.advance(seconds=2)
            await r.next()
        assert svc.slots.open_count() == 1
        with db.begin() as s:  # the slot is gone (e.g. the renewals failed past its expiry and it was reclaimed)
            s.query(StreamSlotRow).delete()
        frames = [f async for f in _drain(r)]
        assert r.closed and not [f for f in frames if f.get("event") in ("end", "reconnect")]  # no further frame

    asyncio.run(go())


async def _drain(r: Run):
    for _ in range(50):
        f = await r.next()
        if f is None:
            return
        yield f


def test_a_stream_writes_only_its_slot_row(db, time_) -> None:
    """Decision A3: the only rows a stream writes are its own bookkeeping (stream_slots); no domain table, no event."""
    written: list[str] = []
    engine = db.kw["bind"]

    def record(_conn, _cursor, statement: str, *_args) -> None:
        verb = statement.lstrip().split(" ", 1)[0].upper()
        if verb in ("INSERT", "UPDATE", "DELETE"):
            written.append(statement)

    async def go() -> None:
        clock = Clock()
        svc = shared_service(db, time_, clock, max_seconds=30, slot_ttl_seconds=3.0)
        r = Run(svc, FakeReader({"task:t1": "t1"}))
        await r.until(lambda f: f.get("event") == "reconnect")
        await r.next()

    orm_event.listen(engine, "before_cursor_execute", record)
    try:
        asyncio.run(go())
    finally:
        orm_event.remove(engine, "before_cursor_execute", record)
    assert written and all("stream_slots" in w for w in written), written


# --- the startup check -----------------------------------------------------------------------------------------------


PG = "postgresql+psycopg://ta@db:5432/ta"


@pytest.mark.parametrize("values, refused", [
    ({}, False),  # SQLite: one process, memory
    ({"database_url": PG}, False),  # PostgreSQL: shared by default
    ({"database_url": PG, "shared_state": "memory"}, True),
    ({"database_url": PG, "shared_state": "memory", "single_process": True}, False),
    ({"database_url": PG, "shared_state": "memory", "single_process": True, "web_concurrency": 2}, True),
    ({"shared_state": "database"}, False),
])
def test_process_local_state_is_refused_where_several_processes_share_postgresql(values, refused) -> None:
    settings = Settings(_env_file=None, **values)
    assert bool(settings.shared_state_problems()) == refused
    if "database_url" in values and "shared_state" not in values:
        assert settings.resolved_shared_state == "database"


def test_web_concurrency_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    settings = Settings(_env_file=None, database_url=PG, shared_state="memory", single_process=True)
    assert settings.web_concurrency == 4 and "WEB_CONCURRENCY=4" in settings.shared_state_problems()[0]


def test_the_container_refuses_to_start_with_process_local_state_on_postgresql() -> None:
    from app.services.container import build_container

    with pytest.raises(ConfigError, match="TA_SHARED_STATE=memory"):
        build_container(Settings(_env_file=None, database_url=PG, shared_state="memory"))


def test_a_production_deployment_refuses_memory_state() -> None:
    settings = Settings(_env_file=None, database_url=PG, shared_state="memory", single_process=True)
    assert any("TA_SHARED_STATE must be 'database'" in p for p in settings.deployment_problems())
