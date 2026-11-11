"""Schema migrations on PostgreSQL (MVP item 8). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to; the role must be allowed to create databases (CI's is
the superuser of a throwaway service).

Each test makes its own database next to the given one and drops it afterwards, so the shared test database (whose
tables the other PostgreSQL tests create with `create_all`) is never stamped or altered. What SQLite cannot show:
the baseline creates PostgreSQL's types, boolean defaults and partial unique indexes exactly as the ORM declares
them; a `create_all` database from before MVP item 2 is adopted (its BOOLEAN and INTEGER columns added with
PostgreSQL's spelling); and the application runs in `migrations` mode on the migrated schema."""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.learner import LearnerProfileInput
from app.services.container import build_container
from app.storage.db import create_db, dispose
from app.storage.migrate import BASELINE, SchemaNotCurrent, current_revision, schema_differences, upgrade
from app.storage.orm import Base
from tests.conftest import LEARNER

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL migration tests")


@pytest.fixture
def fresh_db():
    """The URL of a new, empty database, dropped after the test."""
    name = f"ta_mig_{uuid.uuid4().hex[:12]}"
    admin = create_engine(URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(URL).set(database=name).render_as_string(hide_password=False)
    yield url
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


def state(url: str) -> tuple[str | None, list]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return current_revision(conn), schema_differences(conn)
    finally:
        engine.dispose()


def test_the_baseline_on_postgresql_is_exactly_the_orm_schema(fresh_db) -> None:
    report = upgrade(fresh_db)
    assert (report.after, report.adopted) == (BASELINE, False)
    assert state(fresh_db) == (BASELINE, [])
    engine = create_engine(fresh_db)
    indexes = {i["name"]: i for i in inspect(engine).get_indexes("task_jobs")}
    assert indexes["uq_task_jobs_active"]["unique"]  # the partial unique index the job queue relies on
    with engine.connect() as conn:
        where = conn.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname = 'uq_task_jobs_active'")).scalar()
    assert "WHERE" in where and "QUEUED" in where
    engine.dispose()
    assert upgrade(fresh_db).describe() == f"already at {BASELINE}"


def test_a_create_all_database_from_before_item_2_is_adopted(fresh_db) -> None:
    dispose(create_db(fresh_db))
    engine = create_engine(fresh_db)
    with engine.begin() as conn:
        for table, column in (("learners", "version"), ("curricula", "version"), ("tasks", "pause_requested"),
                              ("tasks", "cancel_requested")):
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
        conn.execute(text("INSERT INTO learners (learner_id, updated_at, body) VALUES ('kept', now(), '{}')"))
    engine.dispose()
    report = upgrade(fresh_db)
    assert report.adopted and state(fresh_db) == (BASELINE, [])
    engine = create_engine(fresh_db)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version FROM learners WHERE learner_id = 'kept'")).scalar() == 1
    engine.dispose()


def test_the_application_runs_in_migrations_mode_on_postgresql(fresh_db, tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, database_url=fresh_db, database_schema="migrations", log_json=False)
    with pytest.raises(SchemaNotCurrent):  # an empty database: the app does not create the schema itself
        build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    engine = create_engine(fresh_db)
    assert inspect(engine).get_table_names() == []
    engine.dispose()

    upgrade(fresh_db)
    c = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    try:
        c.learner_service.upsert(LEARNER["learner_id"], LearnerProfileInput.model_validate(LEARNER["profile"]))
        assert c.learner_service.get(LEARNER["learner_id"]).display_name == LEARNER["profile"]["display_name"]
        task = c.orchestrator.create_task(request=LEARNER["request"], learner_id=LEARNER["learner_id"], user_id="u")
        assert c.task_service.get(task.task_id).task_id == task.task_id
    finally:
        c.close()
    engine = create_engine(fresh_db)
    assert set(inspect(engine).get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
    engine.dispose()
