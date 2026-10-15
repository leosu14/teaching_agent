"""SQLAlchemy engine/session factory. Only the storage package touches SQLAlchemy."""

from __future__ import annotations

from pathlib import Path

import logging

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.storage.orm import Base

logger = logging.getLogger(__name__)

# Columns added after a table first shipped: `create_all` never alters an existing table, so a database created by an
# earlier version gets them here (SQLite; other databases are migrated by their operator).
ADDED_COLUMNS = {
    "tasks": {"version": "INTEGER NOT NULL DEFAULT 1", "lease_owner": "VARCHAR(64)",
              "lease_fence": "INTEGER NOT NULL DEFAULT 0", "lease_expires_ms": "BIGINT",
              "pause_requested": "BOOLEAN NOT NULL DEFAULT 0", "cancel_requested": "BOOLEAN NOT NULL DEFAULT 0"},
    "learning_cycles": {"lease_token": "VARCHAR(64)", "lease_expires_ms": "BIGINT"},
}


def create_db(url: str) -> sessionmaker[Session]:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _sqlite_pragmas)
    Base.metadata.create_all(engine)
    if url.startswith("sqlite"):
        _upgrade_sqlite(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _upgrade_sqlite(engine: Engine) -> None:
    present = inspect(engine)
    with engine.begin() as conn:
        for table, columns in ADDED_COLUMNS.items():
            have = {c["name"] for c in present.get_columns(table)}
            for name, ddl in columns.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
    try:
        with engine.begin() as conn:  # a no-op on a database created with the constraint
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_artifact_version_idx "
                              "ON artifacts (task_id, name, version)"))
    except Exception:  # duplicate versions written before PR18: leave them, new writes are fenced
        logger.warning("artifacts: existing duplicate (task_id, name, version) rows; unique index not created")


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - trivial
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def dispose(factory: sessionmaker[Session]) -> None:
    bind = factory.kw.get("bind")
    if isinstance(bind, Engine):
        bind.dispose()
