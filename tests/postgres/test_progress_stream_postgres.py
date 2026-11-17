"""Issue #54 on PostgreSQL: the progress stream's cursor where commits can be seen out of `seq` order. Opt-in like
the other PostgreSQL tests (TA_TEST_POSTGRES_URL).

- A delayed commit: one transaction inserts an event (its `seq` is allocated) and holds its commit while another
  commits a later-numbered event of the same source. Within the settled bound the stream delivers neither past the
  held row; once the held row commits it delivers both, in `seq` order, on the open stream and after a reconnect.
- The same with the commit held beyond the bound: the documented limit (the event can be passed), recorded here so
  the assumption is tested at its edge, not hidden.
- The two-source reconnect scenario of tests/e2e/test_progress_stream_api.py on a PostgreSQL-backed API."""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from datetime import UTC, datetime

import pytest
from sqlalchemy import insert

from app.schemas.common import new_id
from app.schemas.events import Event, EventType
from app.services.progress_stream import ProgressStreamService, StreamCursor, StreamLimits
from app.storage.db import create_db, dispose
from app.storage.orm import EventRow
from app.storage.repositories import SqlEventRepository
from tests import sse
from tests.unit.test_progress_stream import FakeReader

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL stream tests")

SETTLE = 0.6


@pytest.fixture
def sessions():
    made = create_db(URL)
    yield made
    dispose(made)


def event(task_id: str, name: str) -> Event:
    return Event(event_id=new_id(name), type=EventType.NODE_STARTED, task_id=task_id, at=datetime.now(UTC))


class Consumer:
    """A stream iterated on its own event loop in a thread (real clock), its frames collected as they come."""

    def __init__(self, svc: ProgressStreamService, reader: FakeReader, cursor: dict | None = None) -> None:
        self.frames: queue.Queue = queue.Queue()
        self.got: list[dict] = []
        self._stop = threading.Event()
        slot = svc.slots.acquire("p")
        self._gen = svc._run(reader, dict(cursor or {}), lambda: None, slot)
        self._thread = threading.Thread(target=lambda: asyncio.run(self._consume()), daemon=True)
        self._thread.start()

    async def _consume(self) -> None:
        try:
            async for chunk in self._gen:
                for f in sse.parse(chunk):
                    self.frames.put(f)
                if self._stop.is_set():
                    return
        finally:
            await self._gen.aclose()

    def collect(self, seconds: float) -> list[dict]:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                self.got.append(self.frames.get(timeout=0.02))
            except queue.Empty:
                pass
        return self.got

    def ids(self) -> list[str]:
        return [f["data"]["event"]["event_id"] for f in self.got if f.get("event") == "progress"]

    def cursor(self) -> dict:
        return StreamCursor.decode([f["id"] for f in self.got if "id" in f][-1])

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def service(sessions) -> ProgressStreamService:
    return ProgressStreamService(tasks=None, cycles=None, events=SqlEventRepository(sessions),  # type: ignore[arg-type]
                                 limits=StreamLimits(settle_seconds=SETTLE, poll_seconds=0.03, heartbeat_seconds=60))


def held_insert(sessions, e: Event):
    """Insert `e` in a transaction that stays open (its `seq` allocated, the row not visible) until committed."""
    s = sessions()
    tx = s.begin()
    s.execute(insert(EventRow).values(event_id=e.event_id, task_id=e.task_id, type=e.type, at=e.at,
                                      body=e.model_dump_json()))
    s.flush()
    return s, tx


def seqs(sessions, *events: Event) -> list[int]:
    with sessions() as s:
        return [s.query(EventRow.seq).filter(EventRow.event_id == e.event_id).scalar() for e in events]


def test_a_delayed_commit_within_the_bound_is_delivered_in_order(sessions) -> None:
    svc, task = service(sessions), new_id("t")
    reader = FakeReader({f"task:{task}": task})
    stream = Consumer(svc, reader)
    try:
        stream.collect(0.2)
        late, early = event(task, "late"), event(task, "early")
        s, tx = held_insert(sessions, late)  # lower seq, not committed
        SqlEventRepository(sessions).append(early)  # higher seq, committed first
        low, high = seqs(sessions, late, early)
        assert low is None and high is not None  # the held row is invisible to other sessions
        stream.collect(SETTLE / 2)
        assert stream.ids() == []  # nothing past the held row: `early` has not settled yet
        tx.commit()
        s.close()
        stream.collect(SETTLE * 4)
        assert stream.ids() == [late.event_id, early.event_id]  # both, in seq order
        assert seqs(sessions, late)[0] < seqs(sessions, early)[0]
        last = stream.cursor()
    finally:
        stream.stop()
    # a reconnect from before them gets both; one from the last cursor gets nothing again
    again = Consumer(svc, reader, {f"task:{task}": 0})
    after = Consumer(svc, reader, last)
    try:
        again.collect(SETTLE * 3)
        after.collect(SETTLE * 3)
        assert again.ids() == [late.event_id, early.event_id] and after.ids() == []
    finally:
        again.stop()
        after.stop()


def test_a_commit_held_beyond_the_bound_is_the_documented_limit(sessions) -> None:
    svc, task = service(sessions), new_id("t")
    reader = FakeReader({f"task:{task}": task})
    stream = Consumer(svc, reader)
    try:
        stream.collect(0.2)
        stalled, early = event(task, "stalled"), event(task, "early")
        s, tx = held_insert(sessions, stalled)
        SqlEventRepository(sessions).append(early)
        stream.collect(SETTLE * 4)  # held far beyond the bound: the cursor moves past the stalled row
        assert stream.ids() == [early.event_id]
        tx.commit()
        s.close()
        stream.collect(SETTLE * 3)
        assert stream.ids() == [early.event_id]  # passed: the limit docs/architecture.md documents
        last = stream.cursor()
    finally:
        stream.stop()
    after = Consumer(svc, reader, last)
    try:
        after.collect(SETTLE * 3)
        assert after.ids() == []  # a reconnect from the last cursor does not recover it either
    finally:
        after.stop()
    # It is still stored and served by GET /tasks/{id}/events: the stream is not the only way to read events.
    assert stalled.event_id in {e.event_id for e in SqlEventRepository(sessions).list_for_task(task)}


def test_reconnects_across_both_sources_and_a_step_change_on_postgres(tmp_path) -> None:
    from app.providers.llm.mock import MockLLMProvider
    from app.providers.llm.mock_responders import default_responders
    from app.services.container import build_container
    from scripts.run_learning_cycle_demo import settings
    from tests.auth import TestAuth, client_for
    from tests.e2e.test_progress_stream_api import Api, check_reconnects
    from tests.postgres.test_student_app_postgres import placed_learner

    c = build_container(settings(tmp_path / "data", database_url=URL, log_json=False, stream_settle_seconds=0.3,
                                 stream_poll_seconds=0.02),
                        llm_providers={"mock": MockLLMProvider(default_responders())})
    try:
        learner_id = placed_learner(c)
        auth = TestAuth()
        with client_for(c, learner_id, auth=auth) as a:
            goal = a.post(f"/learners/{learner_id}/goals", json={
                "title": "Reach B1 Spanish", "domain": "spanish", "target_level": "B1",
                "idempotency_key": new_id("goal"), "metadata": {"framework_id": "cefr"}}).json()
            assert a.post(f"/goals/{goal['goal_id']}/curriculum").status_code == 201
            view = a.post(f"/learners/{learner_id}/learning-cycles", json={"idempotency_key": new_id("w")}).json()
        api = Api(c, learner_id, auth=auth)
        try:
            check_reconnects(api, view)
        finally:
            api.close()
    finally:
        c.close()
