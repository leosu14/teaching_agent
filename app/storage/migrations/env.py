"""Alembic environment for the application database (see app/storage/migrate.py, which runs it).

The schema's source of truth is `app.storage.orm.Base.metadata`; each revision in `versions/` moves a database from
one released shape of it to the next. `migrate.py` passes an open connection in `config.attributes["connection"]`;
the URL in `sqlalchemy.url` is used otherwise (the `alembic` command line, for writing a new revision).
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from app.storage.orm import Base

config = context.config
target_metadata = Base.metadata


def _configure(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True,
                      compare_server_default=True,
                      render_as_batch=connection.dialect.name == "sqlite")  # SQLite alters tables by copying them


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _configure(connection)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.connect() as conn:
        _configure(conn)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata,
                      literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    run_migrations_online()
