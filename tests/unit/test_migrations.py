"""Schema migrations (MVP item 8): the Alembic baseline is the ORM's schema, `upgrade` adopts a database that
`create_all` made earlier, and `migrations` mode refuses a database that is not at the newest revision."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from app.storage.db import create_db, dispose
from app.storage.migrate import (
    BASELINE,
    AdoptionRefused,
    SchemaNotCurrent,
    check,
    current_revision,
    head_revision,
    schema_differences,
    upgrade,
)
from app.storage.orm import Base


def url(tmp_path, name="db") -> str:
    return f"sqlite:///{tmp_path / f'{name}.sqlite'}"


def revision(u: str) -> str | None:
    engine = create_engine(u)
    try:
        with engine.connect() as conn:
            return current_revision(conn)
    finally:
        engine.dispose()


def diffs(u: str) -> list:
    engine = create_engine(u)
    try:
        with engine.connect() as conn:
            return schema_differences(conn)
    finally:
        engine.dispose()


def test_one_head_and_the_baseline_is_it() -> None:
    assert head_revision() == BASELINE


def test_upgrade_of_an_empty_database_is_exactly_the_orm_schema(tmp_path) -> None:
    u = url(tmp_path)
    report = upgrade(u)
    assert (report.before, report.after, report.adopted) == (None, BASELINE, False)
    assert revision(u) == BASELINE
    # A model change without a revision fails here: the migrated schema must equal the ORM's, table by table.
    assert diffs(u) == []
    engine = create_engine(u)
    assert set(inspect(engine).get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
    engine.dispose()


def test_upgrade_again_changes_nothing(tmp_path) -> None:
    u = url(tmp_path)
    upgrade(u)
    report = upgrade(u)
    assert report.describe() == f"already at {BASELINE}" and not report.adopted


def test_a_create_all_database_is_adopted_with_its_missing_columns(tmp_path) -> None:
    u = url(tmp_path)
    dispose(create_db(u))  # the "create" mode of every earlier version
    engine = create_engine(u)
    with engine.begin() as conn:  # a database from before MVP item 2: no versioned learner or curriculum
        conn.execute(text("ALTER TABLE learners DROP COLUMN version"))
        conn.execute(text("ALTER TABLE curricula DROP COLUMN version"))
        conn.execute(text("INSERT INTO learners (learner_id, updated_at, body) "
                          "VALUES ('kept', '2026-01-01', '{}')"))
    engine.dispose()
    report = upgrade(u)
    assert report.adopted and report.after == BASELINE
    assert "adopted" in report.describe()
    assert diffs(u) == [] and revision(u) == BASELINE
    engine = create_engine(u)
    with engine.connect() as conn:  # the data is kept, the new column has its default
        assert conn.execute(text("SELECT version FROM learners WHERE learner_id = 'kept'")).scalar() == 1
    engine.dispose()


def test_adoption_refuses_a_database_that_differs_otherwise_and_leaves_it_unstamped(tmp_path) -> None:
    u = url(tmp_path)
    dispose(create_db(u))
    engine = create_engine(u)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_tasks_status"))
        conn.execute(text("ALTER TABLE tasks ADD COLUMN surprise INTEGER"))
    engine.dispose()
    with pytest.raises(AdoptionRefused) as refused:
        upgrade(u)
    assert "ix_tasks_status" in str(refused.value) and "surprise" in str(refused.value)
    assert revision(u) is None


def test_migrations_mode_refuses_a_database_not_at_the_newest_revision(tmp_path) -> None:
    created = url(tmp_path, "created")
    dispose(create_db(created))
    with pytest.raises(SchemaNotCurrent, match="not under schema migrations.*migrate upgrade"):
        create_db(created, "migrations")
    empty = url(tmp_path, "empty")
    with pytest.raises(SchemaNotCurrent):
        create_db(empty, "migrations")
    engine = create_engine(empty)
    assert inspect(engine).get_table_names() == []  # migrations mode never creates anything
    engine.dispose()


def test_migrations_mode_runs_on_a_migrated_database(tmp_path) -> None:
    u = url(tmp_path)
    upgrade(u)
    sessions = create_db(u, "migrations")
    try:
        engine = sessions.kw["bind"]
        assert check(engine) == BASELINE
    finally:
        dispose(sessions)


def test_a_stamped_but_older_revision_is_refused(tmp_path) -> None:
    u = url(tmp_path)
    upgrade(u)
    engine = create_engine(u)
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = '0000_older'"))
    with pytest.raises(SchemaNotCurrent, match="revision 0000_older"):
        check(engine)
    engine.dispose()
