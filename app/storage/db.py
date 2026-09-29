"""SQLAlchemy engine/session factory. Only the storage package touches SQLAlchemy."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.storage.orm import Base


def create_db(url: str) -> sessionmaker[Session]:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _sqlite_pragmas)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - trivial
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def dispose(factory: sessionmaker[Session]) -> None:
    bind = factory.kw.get("bind")
    if isinstance(bind, Engine):
        bind.dispose()
