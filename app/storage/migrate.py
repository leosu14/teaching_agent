"""Schema migrations (Alembic) for deployed databases: `python -m app.storage.migrate upgrade | check | current`.

Two ways to get a schema exist, chosen by `TA_DATABASE_SCHEMA`:

- `create` (development and tests, the default outside a production deployment): `create_db` creates missing tables
  on startup and, on SQLite only, adds the columns of `ADDED_COLUMNS`. Nothing is versioned.
- `migrations` (a production deployment, `TA_DEPLOYMENT=production`): the schema is versioned by the revisions in
  `app/storage/migrations/versions/`. A one-shot `upgrade` brings the database to the newest revision before a new
  version of the API or worker starts, and those processes refuse to start on a database that is not at it
  (`SchemaNotCurrent`), instead of running against columns they do not know.

`upgrade` on a database created earlier by `create_all` (no `alembic_version` table, but the application's tables)
adopts it: the tables and columns the baseline has and it lacks are added, the result is compared with the baseline,
and only an exact match is stamped. A database that differs in any other way is refused, with the differences, and
left as it was.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine

from app.config.routing import ConfigError
from app.storage.orm import Base

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
BASELINE = "0001_baseline"
VERSION_TABLE = "alembic_version"


class SchemaNotCurrent(ConfigError):
    """The database is not at the newest schema revision (or is not under migrations at all)."""


class AdoptionRefused(ConfigError):
    """A database created without migrations differs from the baseline in a way adoption does not repair."""


@dataclass(frozen=True)
class UpgradeReport:
    before: str | None
    after: str
    adopted: bool = False

    def describe(self) -> str:
        if self.adopted:
            then = "" if self.after == BASELINE else f", then upgraded to {self.after}"
            return f"adopted an existing database at {BASELINE}{then}"
        if self.before == self.after:
            return f"already at {self.after}"
        return f"upgraded from {self.before or 'an empty database'} to {self.after}"


def alembic_config(url: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if url:
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def head_revision() -> str:
    heads = ScriptDirectory.from_config(alembic_config()).get_heads()
    if len(heads) != 1:
        raise ConfigError(f"the schema migrations have {len(heads)} heads ({heads}); merge them into one")
    return heads[0]


def current_revision(connection: Connection) -> str | None:
    return MigrationContext.configure(connection).get_current_revision()


def schema_differences(connection: Connection) -> list:
    """Differences between the database and the ORM metadata (an empty list: they match)."""
    context = MigrationContext.configure(connection, opts={"compare_type": True, "compare_server_default": False})
    return compare_metadata(context, Base.metadata)


def check(engine: Engine) -> str:
    """The database's revision when it is the newest one; SchemaNotCurrent otherwise."""
    head = head_revision()
    with engine.connect() as conn:
        current = current_revision(conn)
    if current != head:
        state = "is not under schema migrations" if current is None else f"is at revision {current}"
        raise SchemaNotCurrent(f"the database {state}, this version needs {head}: run "
                               f"`python -m app.storage.migrate upgrade` before starting it")
    return current


def upgrade(url: str) -> UpgradeReport:
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            before = current_revision(conn)
            legacy = before is None and _has_application_tables(conn)
        if legacy:
            _adopt(engine)
            before = BASELINE
        cfg = alembic_config(url)
        with engine.begin() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
        with engine.connect() as conn:
            after = current_revision(conn)
        return UpgradeReport(before=None if legacy else before, after=after or "", adopted=legacy)
    finally:
        engine.dispose()


def _adopt(engine: Engine) -> None:
    """Bring a `create_all` database to the baseline (missing tables, the columns of ADDED_COLUMNS), compare, stamp."""
    from app.storage.db import ADDED_COLUMNS, add_missing_columns

    Base.metadata.create_all(engine)  # tables (with their indexes) the database does not have yet; never alters
    add_missing_columns(engine, ADDED_COLUMNS)
    _drop_redundant_indexes(engine)
    with engine.connect() as conn:
        diffs = [d for d in schema_differences(conn) if not _ignorable(d)]
    if diffs:
        raise AdoptionRefused("the existing database differs from the schema baseline and was not stamped: "
                              + "; ".join(_describe(d) for d in diffs[:20])
                              + (f" (and {len(diffs) - 20} more)" if len(diffs) > 20 else ""))
    cfg = alembic_config(str(engine.url))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.stamp(cfg, BASELINE)
    logger.info("adopted an existing database at schema revision %s", BASELINE)


# Indexes an earlier `create` mode added that the baseline expresses otherwise: SQLite's upgrade path created
# uq_artifact_version_idx for databases older than the uq_artifact_version constraint (the same rule).
EXTRA_INDEXES = {"uq_artifact_version_idx"}


def _drop_redundant_indexes(engine: Engine) -> None:
    """SQLite's `create` mode adds uq_artifact_version_idx to every database; where the table has the constraint it
    duplicates, the index goes (an older table without the constraint keeps it: then it is the rule)."""
    present = inspect(engine)
    if "artifacts" not in present.get_table_names():
        return
    constraints = {c["name"] for c in present.get_unique_constraints("artifacts")}
    indexes = {i["name"] for i in present.get_indexes("artifacts")}
    if "uq_artifact_version" in constraints and "uq_artifact_version_idx" in indexes:
        with engine.begin() as conn:
            conn.execute(text("DROP INDEX uq_artifact_version_idx"))


def _ignorable(diff) -> bool:
    return isinstance(diff, tuple) and diff[0] == "remove_index" and diff[1].name in EXTRA_INDEXES


def _describe(diff) -> str:
    if isinstance(diff, list):
        return ", ".join(_describe(d) for d in diff)
    kind, *rest = diff
    names = [getattr(x, "name", None) or (str(x) if isinstance(x, str) else type(x).__name__) for x in rest[:3]]
    return f"{kind} {' '.join(n for n in names if n)}"


def _has_application_tables(conn: Connection) -> bool:
    present = set(inspect(conn).get_table_names())
    return bool(present & set(Base.metadata.tables))


def _engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point (tested via functions)
    from app.config.secrets import load_secret_files
    from app.config.settings import Settings
    from app.observability.logging import configure_logging

    parser = argparse.ArgumentParser(description="Manage the database schema (Alembic revisions).")
    parser.add_argument("action", choices=["upgrade", "check", "current"],
                        help="upgrade: to the newest revision (adopting a create_all database); check: exit 1 unless "
                             "the database is at the newest revision; current: print its revision")
    args = parser.parse_args(argv)
    load_secret_files()
    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)
    url = settings.resolved_database_url
    try:
        if args.action == "upgrade":
            print(f"schema: {upgrade(url).describe()}")
            return 0
        engine = _engine(url)
        try:
            if args.action == "current":
                with engine.connect() as conn:
                    print(current_revision(conn) or "none")
                print(f"newest: {head_revision()}")
                return 0
            print(f"schema: at {check(engine)}")
            return 0
        finally:
            engine.dispose()
    except ConfigError as exc:
        print(f"PROBLEM: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
