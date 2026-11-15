"""Fixtures of the PostgreSQL tests (opt-in: TA_TEST_POSTGRES_URL)."""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

URL = os.environ.get("TA_TEST_POSTGRES_URL")


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
