"""Shared state of several processes on one database (multi-replica readiness, issue #66 PR 2).

Several API processes and workers on different machines share one PostgreSQL database. What used to be counted in
one process's memory is counted here, in database time (`DbClock`), so that every process makes the same decision:

- `SqlRateLimits`: a provider's requests per minute (a sliding window of admitted starts) and its concurrency
  (slots held as leases), per (capability, provider) key. An admission is decided in one transaction under an advisory
  lock per key (PostgreSQL), with a lock and statement timeout: a slow or unreachable database fails the admission,
  and the caller refuses the paid call (D3, fail closed).
- `SqlStreamSlots`: the open progress streams per principal and in total, as leases renewed while a stream is open.
- `SqlResearchCacheStore`: web search results per learner, for 7 days (D4), erased with the learner's data.

Every slot is a lease: the holder renews it and deletes it when it is done, and one whose process died frees itself
at its expiry (the next admission deletes expired rows before it counts, and the sweeper deletes the rest).

A renew keeps a row alive only while the row still exists. That is safe: a row that expired is deleted by the first
admission that counts after its expiry, so while it still exists nobody was admitted in its place.

These tables are bookkeeping, not domain data: the rate-limit and stream-slot tables are the only ones a GET route
writes (decision A3), and they hold no learner data (a stream slot stores a hash of the principal only).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from app.schemas.common import new_id, utcnow
from app.schemas.lease import learner_writes
from app.storage.dbclock import DbClock
from app.storage.orm import RateLimitSlotRow, RateLimitStartRow, ResearchCacheRow, StreamSlotRow
from app.storage.repositories import _guard, _ms

Clock = Callable[[], datetime]

WINDOW_MS = 60_000  # the requests-per-minute window
# The longest an admission or a stream-slot acquisition may wait on a lock or a statement (PostgreSQL). Past it the
# transaction fails, and the caller refuses the request (fail closed) instead of waiting on a database in trouble.
DECISION_TIMEOUT_MS = 5_000
# The advisory-lock key space of these decisions: the two-key form, apart from the one-key locks (learners, objects)
# and the event-order lock (54, 1).
RATE_LIMIT_LOCK_SPACE = 66
STREAM_SLOTS_LOCK = (66, 2)


def _bounded(s: Session) -> None:
    """Bound the transaction's lock waits and statements (PostgreSQL; SQLite waits `busy_timeout`, 5 s)."""
    if s.get_bind().dialect.name == "postgresql":
        s.execute(text(f"SET LOCAL lock_timeout = {DECISION_TIMEOUT_MS}"))
        s.execute(text(f"SET LOCAL statement_timeout = {DECISION_TIMEOUT_MS}"))


def _lock(s: Session, space: int, key: int) -> None:
    """A transaction-scoped advisory lock (PostgreSQL). SQLite: the caller's first statement is a write, which takes
    the database's write lock for the rest of the transaction."""
    if s.get_bind().dialect.name == "postgresql":
        s.execute(text("SELECT pg_advisory_xact_lock(CAST(:space AS integer), CAST(:key AS integer))"),
                  {"space": space, "key": key})


def _key_lock(key: str) -> int:
    """A stable 32-bit lock id for a rate-limit key (the same in every process)."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big", signed=True)


# --- the rate limiter ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Admission:
    """The decision on one request: admitted (with its concurrency slot, when the key has a concurrency limit), or
    not yet, with the limit that holds it back and, for the window, how long until a start leaves it."""

    admitted: bool
    slot_id: str | None = None
    reason: str | None = None  # "concurrency" | "requests_per_minute" when not admitted
    wait_seconds: float | None = None


class SqlRateLimits:
    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow, *,
                 db_clock: DbClock | None = None) -> None:
        self._sessions = sessions
        self._time = db_clock or DbClock(clock)

    def admit(self, key: str, *, requests_per_minute: int | None, max_concurrency: int | None,
              slot_ttl: timedelta) -> Admission:
        """Admit one request of `key`, or say why not. One transaction: the key's lock, then the expired starts and
        slots deleted (the first statement on SQLite is this write, so it holds the write lock), then the counts and
        the new rows. Raises whatever the database raises (the caller fails closed)."""
        with self._sessions.begin() as s:
            _bounded(s)
            _lock(s, RATE_LIMIT_LOCK_SPACE, _key_lock(key))
            now = _ms(self._time.now(s))
            s.execute(delete(RateLimitStartRow).where(RateLimitStartRow.limit_key == key,
                                                      RateLimitStartRow.started_ms <= now - WINDOW_MS))
            s.execute(delete(RateLimitSlotRow).where(RateLimitSlotRow.limit_key == key,
                                                     RateLimitSlotRow.expires_ms <= now))
            if max_concurrency is not None:
                live = s.scalar(select(func.count()).select_from(RateLimitSlotRow)
                                .where(RateLimitSlotRow.limit_key == key)) or 0
                if live >= max_concurrency:
                    return Admission(admitted=False, reason="concurrency")
            if requests_per_minute is not None:
                count, oldest = s.execute(select(func.count(), func.min(RateLimitStartRow.started_ms))
                                          .where(RateLimitStartRow.limit_key == key)).one()
                if count >= requests_per_minute:
                    wait = max((oldest + WINDOW_MS - now) / 1000, 0.001)
                    return Admission(admitted=False, reason="requests_per_minute", wait_seconds=round(wait, 3))
                s.add(RateLimitStartRow(start_id=new_id("rls"), limit_key=key, started_ms=now))
            slot_id = None
            if max_concurrency is not None:
                slot_id = new_id("rlslot")
                s.add(RateLimitSlotRow(slot_id=slot_id, limit_key=key,
                                       expires_ms=now + int(slot_ttl.total_seconds() * 1000)))
            return Admission(admitted=True, slot_id=slot_id)

    def renew(self, slot_id: str, ttl: timedelta) -> bool:
        """Extend a held slot to database time + `ttl`. False when the slot is gone (it expired and was deleted)."""
        with self._sessions.begin() as s:
            _bounded(s)
            now = _ms(self._time.now(s))
            return s.execute(update(RateLimitSlotRow).where(RateLimitSlotRow.slot_id == slot_id)
                             .values(expires_ms=now + int(ttl.total_seconds() * 1000))).rowcount == 1

    def release(self, slot_id: str) -> None:
        with self._sessions.begin() as s:
            _bounded(s)
            s.execute(delete(RateLimitSlotRow).where(RateLimitSlotRow.slot_id == slot_id))

    def in_flight(self, key: str) -> int:
        """The key's live slots (tests and operators)."""
        with self._sessions() as s:
            now = _ms(self._time.now(s))
            return s.scalar(select(func.count()).select_from(RateLimitSlotRow).where(
                RateLimitSlotRow.limit_key == key, RateLimitSlotRow.expires_ms > now)) or 0

    def purge_expired(self) -> int:
        """The sweeper's step: delete every expired start and slot (of keys no admission has counted since)."""
        with self._sessions.begin() as s:
            now = _ms(self._time.now(s))
            starts = s.execute(delete(RateLimitStartRow).where(RateLimitStartRow.started_ms <= now - WINDOW_MS))
            slots = s.execute(delete(RateLimitSlotRow).where(RateLimitSlotRow.expires_ms <= now))
            return (starts.rowcount or 0) + (slots.rowcount or 0)


# --- progress-stream slots -----------------------------------------------------------------------------------------


def principal_hash(principal: str) -> str:
    """What a stream slot stores of its principal: a SHA-256, never the identity itself."""
    return hashlib.sha256(principal.encode("utf-8")).hexdigest()


class SqlStreamSlots:
    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow, *,
                 db_clock: DbClock | None = None) -> None:
        self._sessions = sessions
        self._time = db_clock or DbClock(clock)

    def acquire(self, principal: str, *, per_principal: int, total: int, ttl: timedelta) -> str | None:
        """A slot for one more stream of `principal`, or None when the principal or the deployment has as many open
        streams as allowed. One transaction under the slots' lock: expired slots deleted (the first write on SQLite),
        then both counts, then the new slot. Raises whatever the database raises."""
        owner = principal_hash(principal)
        with self._sessions.begin() as s:
            _bounded(s)
            _lock(s, *STREAM_SLOTS_LOCK)
            now = _ms(self._time.now(s))
            s.execute(delete(StreamSlotRow).where(StreamSlotRow.expires_ms <= now))
            open_total = s.scalar(select(func.count()).select_from(StreamSlotRow)) or 0
            mine = s.scalar(select(func.count()).select_from(StreamSlotRow)
                            .where(StreamSlotRow.principal_hash == owner)) or 0
            if open_total >= total or mine >= per_principal:
                return None
            slot_id = new_id("sslot")
            s.add(StreamSlotRow(slot_id=slot_id, principal_hash=owner,
                                expires_ms=now + int(ttl.total_seconds() * 1000)))
            return slot_id

    def renew(self, slot_id: str, ttl: timedelta) -> bool:
        """Extend a held slot to database time + `ttl`. False when it is gone (expired and deleted)."""
        with self._sessions.begin() as s:
            _bounded(s)
            now = _ms(self._time.now(s))
            return s.execute(update(StreamSlotRow).where(StreamSlotRow.slot_id == slot_id)
                             .values(expires_ms=now + int(ttl.total_seconds() * 1000))).rowcount == 1

    def release(self, slot_id: str) -> None:
        with self._sessions.begin() as s:
            _bounded(s)
            s.execute(delete(StreamSlotRow).where(StreamSlotRow.slot_id == slot_id))

    def open_count(self, principal: str | None = None) -> int:
        with self._sessions() as s:
            now = _ms(self._time.now(s))
            query = select(func.count()).select_from(StreamSlotRow).where(StreamSlotRow.expires_ms > now)
            if principal is not None:
                query = query.where(StreamSlotRow.principal_hash == principal_hash(principal))
            return s.scalar(query) or 0

    def purge_expired(self) -> int:
        with self._sessions.begin() as s:
            now = _ms(self._time.now(s))
            return s.execute(delete(StreamSlotRow).where(StreamSlotRow.expires_ms <= now)).rowcount or 0


# --- the research cache --------------------------------------------------------------------------------------------


class SqlResearchCacheStore:
    """Cached search results per (learner, key), as the JSON the research cache writes. Expiry in database time."""

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utcnow, *,
                 db_clock: DbClock | None = None) -> None:
        self._sessions = sessions
        self._time = db_clock or DbClock(clock)

    def get(self, learner_id: str, key: str) -> str | None:
        with self._sessions() as s:
            now = _ms(self._time.now(s))
            return s.scalar(select(ResearchCacheRow.body).where(
                ResearchCacheRow.learner_id == learner_id, ResearchCacheRow.cache_key == key,
                ResearchCacheRow.expires_ms > now))

    def put(self, learner_id: str, key: str, body: str, *, ttl: timedelta, task_id: str) -> None:
        """Store (or replace) an entry until database time + `ttl`. A write for the learner, made for the lesson task
        `task_id`: refused (LearnerDataGone, nothing stored) once that task or the learner's data is deleted, so a
        deletion never leaves an entry behind (the guard takes the deletion lock shared, as every learner write)."""
        with learner_writes(learner_id, task_id, profile=False):
            with self._sessions.begin() as s:
                _guard(s)
                now = _ms(self._time.now(s))
                values = {"learner_id": learner_id, "cache_key": key, "body": body, "created_ms": now,
                          "expires_ms": now + int(ttl.total_seconds() * 1000)}
                insert = postgresql_insert if s.get_bind().dialect.name == "postgresql" else sqlite_insert
                statement = insert(ResearchCacheRow).values(**values)
                s.execute(statement.on_conflict_do_update(
                    index_elements=["learner_id", "cache_key"],
                    set_={"body": statement.excluded.body, "created_ms": statement.excluded.created_ms,
                          "expires_ms": statement.excluded.expires_ms}))

    def delete(self, learner_id: str, key: str) -> None:
        with self._sessions.begin() as s:
            s.execute(delete(ResearchCacheRow).where(ResearchCacheRow.learner_id == learner_id,
                                                     ResearchCacheRow.cache_key == key))

    def entries(self, learner_id: str) -> int:
        """The learner's stored entries, live or expired (tests)."""
        with self._sessions() as s:
            return s.scalar(select(func.count()).select_from(ResearchCacheRow)
                            .where(ResearchCacheRow.learner_id == learner_id)) or 0

    def purge_expired(self, *, limit: int = 1000) -> int:
        """The sweeper's step: delete up to `limit` expired entries."""
        with self._sessions.begin() as s:
            now = _ms(self._time.now(s))
            expired = select(ResearchCacheRow.learner_id, ResearchCacheRow.cache_key).where(
                ResearchCacheRow.expires_ms <= now).limit(limit)
            rows = s.execute(expired).all()
            for learner_id, key in rows:
                s.execute(delete(ResearchCacheRow).where(ResearchCacheRow.learner_id == learner_id,
                                                         ResearchCacheRow.cache_key == key))
            return len(rows)


@dataclass
class SharedStateSweep:
    """What one sweeper pass deleted of the shared state's expired rows."""

    rate_limit_rows: int = 0
    stream_slots: int = 0
    research_cache_entries: int = 0


class SharedStateSweeper:
    def __init__(self, rate_limits: SqlRateLimits, stream_slots: SqlStreamSlots,
                 research_cache: SqlResearchCacheStore) -> None:
        self._rate_limits = rate_limits
        self._stream_slots = stream_slots
        self._research_cache = research_cache

    def sweep(self) -> SharedStateSweep:
        return SharedStateSweep(rate_limit_rows=self._rate_limits.purge_expired(),
                                stream_slots=self._stream_slots.purge_expired(),
                                research_cache_entries=self._research_cache.purge_expired())
