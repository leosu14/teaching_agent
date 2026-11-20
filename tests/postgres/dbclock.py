"""The database-clock fake of the PostgreSQL tests: the database's own `clock_timestamp()`, moved by `offset`.

Lease, fence and spend-expiry decisions on PostgreSQL read the database's clock (app/storage/dbclock.py), so a test
can no longer move them by injecting a process clock. A repository given this fake (`db_clock=`) sees database time
plus `offset`: a test lets a lease run out by moving `offset` forward, as it once moved the injected process clock.
Every reading still comes from the database, inside the statement's own transaction."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.storage.dbclock import DbClock


class ShiftedDbClock(DbClock):
    def __init__(self, offset: timedelta = timedelta(0)) -> None:
        super().__init__()
        self.offset = offset

    def _database(self, s: Session) -> datetime:
        return s.scalar(text("SELECT clock_timestamp() + make_interval(secs => :secs)"),
                        {"secs": self.offset.total_seconds()})


class PinnedDbClock(DbClock):
    """Database time pinned at `at` (read back through the database): a test crosses midnight UTC in database time by
    setting it, as it once set the injected process clock."""

    def __init__(self, at: datetime | None = None) -> None:
        super().__init__()
        self.at = at

    def _database(self, s: Session) -> datetime:
        if self.at is None:
            return s.scalar(text("SELECT clock_timestamp()"))
        return s.scalar(text("SELECT CAST(:at AS timestamptz)"), {"at": self.at})
