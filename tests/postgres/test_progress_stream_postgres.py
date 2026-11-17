"""Issue #54 on PostgreSQL: event commits and the progress stream's cursor. Opt-in like the other PostgreSQL tests
(TA_TEST_POSTGRES_URL).

`seq` is allocated at INSERT and visible at commit, so without an order a lower `seq` could become visible after a
higher one, after a stream's cursor had passed it. `SqlEventRepository.append` takes the event-order lock
(`_lock_event_order`) before its INSERT, so appends commit in `seq` order. A held append is a real `append` stopped
after its INSERT (its `seq` allocated, the lock held) until the test releases it:
- A held append keeps a later append from allocating and committing ahead of it, far beyond the settled bound; once
  it commits, both are visible and streamed in `seq` order, on the open stream and after reconnects.
- A held append that rolls back frees the lock: the later append goes ahead and nothing waits for the gap.
- A stable event id appended twice at once is stored once, without an error.
- Concurrent appends from several writers: no read ever sees a row while a lower one of the same writers is still to
  come, and a stream delivers every event once, in order.
- The two-source reconnect scenario of tests/e2e/test_progress_stream_api.py on a PostgreSQL-backed API.

Each of the first four fails without the lock."""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from datetime import UTC, datetime

import pytest
from sqlalchemy import event as orm_event
from sqlalchemy import select

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


class HeldAppend:
    """`SqlEventRepository.append(e)` in a thread, stopped after its INSERT (inside its transaction: the `seq`
    allocated, the event-order lock held, the row not yet visible) until `release()`; with `rollback=True` it then
    fails there and its transaction rolls back."""

    def __init__(self, sessions, e: Event, *, rollback: bool = False) -> None:
        self.flushed = threading.Event()
        self._release = threading.Event()
        self.error: BaseException | None = None
        self._sessions = sessions
        held = {"done": False}

        def hold(session, _ctx) -> None:
            if held["done"] or not any(isinstance(o, EventRow) and o.event_id == e.event_id for o in session.new):
                return
            held["done"] = True
            self.flushed.set()
            assert self._release.wait(30), "never released"
            if rollback:
                raise RuntimeError("rolled back on purpose")

        self._hook = hold
        orm_event.listen(sessions, "after_flush", hold)

        def run() -> None:
            try:
                SqlEventRepository(sessions).append(e)
            except BaseException as exc:  # noqa: BLE001 - recorded for the test
                self.error = exc

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert self.flushed.wait(10), "the held append never reached its INSERT"

    def release(self) -> None:
        self._release.set()
        self._thread.join(10)
        orm_event.remove(self._sessions, "after_flush", self._hook)


class Append:
    """`SqlEventRepository.append(e)` in a thread, to see whether it is waiting."""

    def __init__(self, sessions, e: Event) -> None:
        self.error: BaseException | None = None

        def run() -> None:
            try:
                SqlEventRepository(sessions).append(e)
            except BaseException as exc:  # noqa: BLE001 - recorded for the test
                self.error = exc

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def waiting(self, seconds: float) -> bool:
        self._thread.join(seconds)
        return self._thread.is_alive()

    def join(self) -> None:
        self._thread.join(10)
        assert not self._thread.is_alive()


def seqs(sessions, *events: Event) -> list[int | None]:
    with sessions() as s:
        return [s.scalar(select(EventRow.seq).where(EventRow.event_id == e.event_id)) for e in events]


def test_a_held_append_keeps_a_later_one_from_committing_ahead_of_it(sessions) -> None:
    svc, task = service(sessions), new_id("t")
    reader = FakeReader({f"task:{task}": task})
    stream = Consumer(svc, reader)
    try:
        stream.collect(0.2)
        stalled, later = event(task, "stalled"), event(task, "later")
        held = HeldAppend(sessions, stalled)
        try:
            second = Append(sessions, later)
            # Far beyond the settled bound: the later append waits for the lock, nothing is visible or streamed.
            assert second.waiting(SETTLE * 4), "a later append committed while an earlier one was held"
            assert seqs(sessions, stalled, later) == [None, None]
            stream.collect(0.1)
            assert stream.ids() == []
        finally:
            held.release()
        second.join()
        assert held.error is None and second.error is None
        low, high = seqs(sessions, stalled, later)
        assert low is not None and high is not None and low < high
        stream.collect(SETTLE * 4)
        assert stream.ids() == [stalled.event_id, later.event_id]  # both, in seq order
        last = stream.cursor()
    finally:
        stream.stop()
    again = Consumer(svc, reader, {f"task:{task}": 0})
    after = Consumer(svc, reader, last)
    try:
        again.collect(SETTLE * 3)
        after.collect(SETTLE * 3)
        assert again.ids() == [stalled.event_id, later.event_id] and after.ids() == []
    finally:
        again.stop()
        after.stop()


def test_a_held_append_that_rolls_back_frees_the_later_one(sessions) -> None:
    svc, task = service(sessions), new_id("t")
    reader = FakeReader({f"task:{task}": task})
    stream = Consumer(svc, reader)
    try:
        stream.collect(0.2)
        doomed, later, third = event(task, "doomed"), event(task, "later"), event(task, "third")
        held = HeldAppend(sessions, doomed, rollback=True)
        try:
            second = Append(sessions, later)
            assert second.waiting(SETTLE * 2), "a later append committed while an earlier one was held"
        finally:
            held.release()
        second.join()
        assert isinstance(held.error, RuntimeError) and second.error is None
        assert seqs(sessions, doomed)[0] is None  # rolled back: its seq is a gap that never fills
        SqlEventRepository(sessions).append(third)
        assert seqs(sessions, later)[0] < seqs(sessions, third)[0]
        stream.collect(SETTLE * 4)
        assert stream.ids() == [later.event_id, third.event_id]
    finally:
        stream.stop()


def test_a_stable_event_id_appended_twice_at_once_is_stored_once(sessions) -> None:
    task = new_id("t")
    e = event(task, "twice")
    held = HeldAppend(sessions, e)
    try:
        again = Append(sessions, e)
        assert again.waiting(0.5)
    finally:
        held.release()
    again.join()
    assert held.error is None and again.error is None  # the second sees the first under the lock: no IntegrityError
    assert [x.event_id for x in SqlEventRepository(sessions).list_for_task(task)] == [e.event_id]


def test_concurrent_appends_become_visible_in_seq_order(sessions) -> None:
    svc = service(sessions)
    tasks = [new_id("t") for _ in range(3)]
    reader = FakeReader({f"task:{t}": t for t in tasks})
    stream = Consumer(svc, reader)
    writers, per_writer = 6, 60
    reads: list[set[tuple[int, str]]] = []
    done = threading.Event()

    def write(n: int) -> None:
        repo = SqlEventRepository(sessions)
        for i in range(per_writer):
            repo.append(event(tasks[(n + i) % len(tasks)], f"w{n}"))

    def read() -> None:
        while not done.is_set():
            with sessions() as s:
                reads.append(set(s.execute(select(EventRow.seq, EventRow.event_id)
                                           .where(EventRow.task_id.in_(tasks))).all()))

    try:
        stream.collect(0.2)
        threads = [threading.Thread(target=write, args=(n,)) for n in range(writers)]
        readers = [threading.Thread(target=read) for _ in range(2)]
        for t in readers + threads:
            t.start()
        for t in threads:
            t.join(120)
        done.set()
        for t in readers:
            t.join(10)
        with sessions() as s:
            final = set(s.execute(select(EventRow.seq, EventRow.event_id).where(EventRow.task_id.in_(tasks))).all())
        assert len(final) == writers * per_writer
        # Every read is a prefix of the final rows: no row was visible while a lower-numbered one was still to come.
        gaps = [r for r in reads if r and r != {x for x in final if x[0] <= max(r)[0]}]
        assert len(reads) > 10 and gaps == []
        deadline = time.monotonic() + 30
        while len(stream.ids()) < len(final) and time.monotonic() < deadline:
            stream.collect(0.2)
        assert stream.ids() == [eid for _, eid in sorted(final)]  # every event once, in seq order
    finally:
        done.set()
        stream.stop()


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
