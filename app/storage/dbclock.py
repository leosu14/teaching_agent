"""The time source of every lease, fence and spend-expiry decision (multi-replica readiness, issue #66 PR 1).

Several API processes and workers on different machines share one PostgreSQL database. Their process clocks drift
apart, so a lease judged by the claimer's clock could be taken over early by a process whose clock runs ahead, and a
reservation could be abandoned before its call ended. Every such decision is therefore made against one clock: the
database's.

- PostgreSQL: `clock_timestamp()`, read inside the same transaction as the conditional statement it guards (a claim,
  a renew, a fenced write, an expiry sweep). `clock_timestamp()` rather than `now()`: `now()` is the transaction's
  start, which a transaction that waited on a row lock would see as stale.
- SQLite (one machine, one clock) and the in-memory stores of unit tests: the injected clock, so tests that move time
  by injecting a clock keep doing so.

A fencing token (the task's `lease_fence`, the cycle's and the teaching session's lease token) stays the safety
mechanism: every protected write still checks the token and the expiry in the same conditional statement. Database
time only makes sure that expiry, and so takeover, means the same instant for every process.

What stays on process clocks: monotonic timers that decide *when* to renew a lease (never *whether* it is held), and
timestamps that are only recorded (created_at, updated_at).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from app.schemas.common import utcnow

Clock = Callable[[], datetime]


class DbClock:
    """Database time on PostgreSQL, the injected `fallback` clock elsewhere."""

    def __init__(self, fallback: Clock = utcnow) -> None:
        self._fallback = fallback

    def now(self, s: Session) -> datetime:
        """The time of a decision made in `s`'s transaction."""
        found = self.database_now(s)
        return found if found is not None else self._fallback()

    def database_now(self, s: Session) -> datetime | None:
        """The database's clock (PostgreSQL), else None: the caller's own clock decides then."""
        if s.get_bind().dialect.name != "postgresql":
            return None
        return _utc(self._database(s))

    def read(self, sessions: sessionmaker[Session]) -> datetime | None:
        """`database_now` in a transaction of its own, for a caller that computes a decision before it writes (the
        write's conditional statement then re-checks against the database's clock)."""
        with sessions() as s:
            return self.database_now(s)

    def _database(self, s: Session) -> datetime:
        return s.scalar(text("SELECT clock_timestamp()"))


def _utc(at: datetime) -> datetime:
    return at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at.astimezone(timezone.utc)
