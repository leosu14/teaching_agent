"""SQLAlchemy engine/session factory. Only the storage package touches SQLAlchemy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.storage.orm import Base

logger = logging.getLogger(__name__)

# Columns added after a table first shipped, before schema migrations existed: `create_all` never alters an existing
# table, so a database created by an earlier version gets them here (in `create` mode on SQLite, and when
# `app.storage.migrate` adopts a database). Later schema changes are Alembic revisions (app/storage/migrations/).
ADDED_COLUMNS = {
    "tasks": {"version": "INTEGER NOT NULL DEFAULT 1", "lease_owner": "VARCHAR(64)",
              "lease_fence": "INTEGER NOT NULL DEFAULT 0", "lease_expires_ms": "BIGINT",
              "pause_requested": "BOOLEAN NOT NULL DEFAULT 0", "cancel_requested": "BOOLEAN NOT NULL DEFAULT 0"},
    "learning_cycles": {"lease_token": "VARCHAR(64)", "lease_expires_ms": "BIGINT"},
    "learners": {"version": "INTEGER NOT NULL DEFAULT 1"},
    "curricula": {"version": "INTEGER NOT NULL DEFAULT 1"},
}


SchemaMode = Literal["create", "migrations"]


def create_db(url: str, schema: SchemaMode = "create") -> sessionmaker[Session]:
    """A session factory. `schema="create"` creates missing tables (and, on SQLite, the ADDED_COLUMNS); `"migrations"`
    changes nothing and refuses a database that is not at the newest Alembic revision (`SchemaNotCurrent`)."""
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
                           pool_pre_ping=not url.startswith("sqlite"))  # a managed database drops idle connections
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _sqlite_pragmas)
    if schema == "migrations":
        from app.storage.migrate import check

        try:
            check(engine)
        except Exception:
            engine.dispose()
            raise
    else:
        Base.metadata.create_all(engine)
        if url.startswith("sqlite"):
            _upgrade_sqlite(engine)
    return sessionmaker(engine, expire_on_commit=False)


def add_missing_columns(engine: Engine, columns: dict[str, dict[str, str]]) -> list[str]:
    """Add each listed column a table lacks (any database). Returns the `table.column` names added."""
    added = []
    present = inspect(engine)
    tables = set(present.get_table_names())
    with engine.begin() as conn:
        for table, specs in columns.items():
            if table not in tables:
                continue
            have = {c["name"] for c in present.get_columns(table)}
            for name, ddl in specs.items():
                if name not in have:
                    if engine.dialect.name != "sqlite":  # BOOLEAN DEFAULT 0 is SQLite's spelling of false
                        ddl = ddl.replace("BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT false")
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    added.append(f"{table}.{name}")
    return added


def _upgrade_sqlite(engine: Engine) -> None:
    add_missing_columns(engine, ADDED_COLUMNS)
    try:
        with engine.begin() as conn:  # a no-op on a database created with the constraint
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_artifact_version_idx "
                              "ON artifacts (task_id, name, version)"))
    except Exception:  # duplicate versions written before PR18: leave them, new writes are fenced
        logger.warning("artifacts: existing duplicate (task_id, name, version) rows; unique index not created")


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - trivial
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")  # a writer waits for another (the spend counters serialize on it)
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def dispose(factory: sessionmaker[Session]) -> None:
    bind = factory.kw.get("bind")
    if isinstance(bind, Engine):
        bind.dispose()
