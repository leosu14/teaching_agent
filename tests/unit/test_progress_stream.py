"""The progress stream service (issue #54): cursor, stop states, settled reads, replay, access loss, heartbeat,
lifetime and limits, on a fake clock with fake readers and an in-memory event table whose rows can be committed out
of `seq` order (as PostgreSQL can)."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from app.curriculum import cycle as engine
from app.schemas.events import Event, EventType
from app.schemas.job import JobKind, JobStatus, TaskJob
from app.schemas.learning_cycle import CycleOutcome, CycleResult, CycleStatus, LearningCycleView
from app.schemas.task import Task, TaskStatus, TaskView, WaitRequest
from app.services.progress_stream import (
    HEARTBEAT,
    InvalidCursor,
    ProgressStreamService,
    Snapshot,
    StreamCursor,
    StreamLimits,
    StreamSlots,
    TooManyStreams,
    _Reader,
    cycle_stopped,
    task_stopped,
)
from tests.unit.test_learning_cycle import T0, action, at, started

# --- the cursor ----------------------------------------------------------------------------------------------------


def test_cursor_round_trips_and_is_versioned() -> None:
    positions = {"cycle": 1042, "task:t_abc": 1077}
    cursor = StreamCursor.encode(positions)
    assert cursor.startswith("v1.") and "=" not in cursor
    assert StreamCursor.decode(cursor) == positions
    assert StreamCursor.decode(None) == {} and StreamCursor.decode("  ") == {}


def _b64(raw: str) -> str:
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


@pytest.mark.parametrize("bad", [
    "v2." + _b64('{"cycle": 1}'), "v1.", "v1.!!!", "v1." + _b64("[1, 2]"), "v1." + _b64('{"cycle": -1}'),
    "v1." + _b64('{"cycle": true}'), "v1." + _b64('{"cycle": 1.5}'), "v1." + _b64('{"other": 1}'),
    "v1." + _b64('{"task:a b": 1}'), "v1." + _b64("not json"), "v1." + "A" * 5000, "garbage",
    "v1." + _b64(json.dumps({f"task:t{i}": 1 for i in range(65)})),
])
def test_a_cursor_that_does_not_parse_is_refused(bad: str) -> None:
    with pytest.raises(InvalidCursor):
        StreamCursor.decode(bad)


# --- stop states -----------------------------------------------------------------------------------------------------


def task(status: TaskStatus, wait: str | None = None) -> Task:
    return Task(task_id="t1", user_id="u", learner_id="l1", request="r", status=status,
                waiting=WaitRequest(node_id="n", kind=wait, prompt={}) if wait else None)


def job(status: JobStatus | None) -> TaskJob | None:
    return None if status is None else TaskJob(job_id="j", task_id="t1", kind=JobKind.SUBMIT_INPUT, status=status,
                                                dedupe_key="d")


@pytest.mark.parametrize("job_status", [None, *JobStatus])
@pytest.mark.parametrize("status,wait", [(s, None) for s in TaskStatus] + [
    (TaskStatus.WAITING, "diagnostic_answers"), (TaskStatus.WAITING, "assessment_answers"),
    (TaskStatus.WAITING, "curriculum_input")])
def test_the_task_stop_table(status: TaskStatus, wait: str | None, job_status: JobStatus | None) -> None:
    active = job_status in (JobStatus.QUEUED, JobStatus.RUNNING)
    stop_status = status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.PAUSED) or (
        status == TaskStatus.WAITING and wait in ("diagnostic_answers", "assessment_answers"))
    assert task_stopped(task(status, wait), job(job_status)) is (stop_status and not active)


def test_named_task_cases() -> None:
    assert not task_stopped(task(TaskStatus.RUNNING), None)  # run inline by a cycle, or crashed awaiting RECOVER
    assert not task_stopped(task(TaskStatus.WAITING, "curriculum_input"), None)  # not waiting for the learner
    assert not task_stopped(task(TaskStatus.FAILED), job(JobStatus.QUEUED))  # retrying with backoff
    assert task_stopped(task(TaskStatus.WAITING, "diagnostic_answers"), job(JobStatus.DONE))


def view(status: CycleStatus, version: int = 5) -> LearningCycleView:
    return LearningCycleView(cycle_id="c", learner_id="l", idempotency_key="k", status=status, action=action(),
                             steps=[], version=version, created_at=T0, updated_at=T0)


@pytest.mark.parametrize("status", list(CycleStatus))
@pytest.mark.parametrize("lease_live", [False, True])
@pytest.mark.parametrize("child_busy", [False, True])
@pytest.mark.parametrize("after_version", [None, 4, 5, 6])
def test_the_cycle_stop_table(status: CycleStatus, lease_live: bool, child_busy: bool,
                              after_version: int | None) -> None:
    stopped = cycle_stopped(view(status, 5), lease_live=lease_live, child_busy=child_busy,
                            after_version=after_version)
    in_progress = status == CycleStatus.RUNNING or lease_live or child_busy
    if in_progress:
        assert not stopped  # in particular RUNNING and not busy never ends the stream
    elif status in (CycleStatus.COMPLETED, CycleStatus.FAILED, CycleStatus.CANCELLED):
        assert stopped
    else:  # WAITING or BLOCKED: only at a version above the client's
        assert stopped is (after_version is None or after_version < 5)


# --- the cycle version advances with every stored change ------------------------------------------------------------


def test_every_cycle_transition_raises_the_version() -> None:
    c = started()
    seen = [c.version]

    def step(change):
        assert change is not None and change.version > seen[-1]
        seen.append(change.version)
        return change

    c = step(engine.published(c, {e.event_id for e in c.pending_events}, T0))
    c = step(engine.claim(c, at(0), at(10)))
    token = c.lease.token
    c = step(engine.renew(c, token, at(6), at(16)))
    c = step(engine.record_artifact(c, "cycle", "art_1", at(1)))
    lesson, session = c.steps
    c = step(engine.begin_step(c, lesson.step_id, at(1)))
    c = step(engine.attach_child(c, lesson.step_id, "task_1", at(2)))
    c = step(engine.observe(c, lesson.step_id, engine.Observation("waiting", ("diagnostic_answers", "n:1")), at(3)))
    c = step(engine.receive(c, at(4)))
    c = step(engine.fail(c, engine.CycleFailure(kind="PROVIDER", retryable=True, message="x",
                                                step_id=lesson.step_id, at=T0), at(5)))
    c = step(engine.resume(c, at(6)))
    c = step(engine.observe(c, lesson.step_id, engine.Observation("completed"), at(7)))
    c = step(engine.release(c, token, at(8)))
    c = step(engine.claim(c, at(20), at(30)))  # a takeover after expiry is a change too
    c = step(engine.begin_step(c, session.step_id, at(21)))
    c = step(engine.attach_child(c, session.step_id, "s_1", at(21)))
    c = step(engine.observe(c, session.step_id, engine.Observation("completed"), at(22)))
    c = step(engine.complete(c, CycleOutcome(result=CycleResult.TAUGHT), at(23)))
    other = started(key="k2")
    cancelled = engine.cancel(other, at(1))
    assert cancelled.version > other.version and cancelled.status == CycleStatus.CANCELLED


# --- limits ----------------------------------------------------------------------------------------------------------


def test_streams_are_capped_per_principal_and_in_total() -> None:
    slots = StreamSlots(per_principal=2, total=3)
    a1, a2 = slots.acquire("a"), slots.acquire("a")
    with pytest.raises(TooManyStreams):
        slots.acquire("a")
    b1 = slots.acquire("b")
    with pytest.raises(TooManyStreams):
        slots.acquire("c")  # the process-wide cap
    a1.release()
    a1.release()  # idempotent: a slot is freed once
    assert slots.open_count("a") == 1 and slots.open_count() == 2
    slots.acquire("c")
    a2.release()
    b1.release()


# --- the loop, on a fake clock ------------------------------------------------------------------------------------


@dataclass
class Row:
    seq: int
    event: Event
    visible_at: float = 0.0  # when its transaction commits (on the fake clock)


class FakeEvents:
    """The events table, with rows whose `seq` is allocated before they commit (become visible)."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self.rows: list[Row] = []
        self.asked: list[set[str]] = []

    def insert(self, task_id: str) -> Row:
        seq = (max((r.seq for r in self.rows), default=0)) + 1
        row = Row(seq, Event(event_id=f"ev{seq}", type=EventType.NODE_STARTED, task_id=task_id,
                             at=datetime(2026, 11, 17, tzinfo=UTC), data={"secret": "ANSWER-KEY"}))
        self.rows.append(row)
        return row

    def _visible(self) -> list[Row]:
        return sorted((r for r in self.rows if r.visible_at <= self.clock.now), key=lambda r: r.seq)

    def max_seq(self) -> int:
        return max((r.seq for r in self._visible()), default=0)

    def after(self, positions: dict[str, int], upto: int, *, limit: int = 500):
        self.asked.append(set(positions))
        out = [(r.seq, r.event) for r in self._visible() if r.seq <= upto
               and r.event.task_id in positions and r.seq > positions[r.event.task_id]]
        return out[:limit]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


class FakeReader(_Reader):
    def __init__(self, sources: dict[str, str]) -> None:
        self.sources = sources
        self.stopped = False
        self.status = TaskStatus.RUNNING
        self.fail: Exception | None = None

    def read(self) -> Snapshot:
        if self.fail is not None:
            raise self.fail
        t = task(self.status).model_copy(update={"created_at": T0, "updated_at": T0})
        return Snapshot(view=TaskView.of(t), stopped=self.stopped, sources=dict(self.sources))


def service(events: FakeEvents, clock: Clock, **limits) -> ProgressStreamService:
    return ProgressStreamService(tasks=None, cycles=None, events=events,  # type: ignore[arg-type]
                                 limits=StreamLimits(**limits), clock=clock, sleep=clock.sleep)


def parse(chunk: str) -> dict:
    out: dict = {}
    for line in chunk.strip("\n").split("\n"):
        key, _, value = line.partition(": ")
        out[key] = value
    return out


class Run:
    """Drive one stream frame by frame; `step` runs until the next frame (or the end)."""

    def __init__(self, svc: ProgressStreamService, reader: FakeReader, cursor: dict | None = None,
                 access=lambda: None) -> None:
        slot = svc.slots.acquire("p")
        self.gen = svc._run(reader, dict(cursor or {}), access, slot)
        self.frames: list[dict] = []
        self.closed = False

    async def next(self) -> dict | None:
        try:
            chunk = await self.gen.__anext__()
        except StopAsyncIteration:
            self.closed = True
            return None
        f = {"comment": True} if chunk == HEARTBEAT else parse(chunk)
        self.frames.append(f)
        return f

    async def until(self, predicate, limit: int = 200) -> dict:
        for _ in range(limit):
            f = await self.next()
            if f is None or predicate(f):
                return f
        raise AssertionError("no such frame")

    def progress(self) -> list[dict]:
        return [json.loads(f["data"]) for f in self.frames if f.get("event") == "progress"]


def run(coro):
    return asyncio.run(coro)


def test_status_first_then_settled_progress_in_seq_order_then_end() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        for key in ("t1", "c1", "t1"):
            events.insert(key)
        events.insert("someone-else")
        reader = FakeReader({"task:t1": "t1", "cycle": "c1"})
        r = Run(service(events, clock), reader)
        first = await r.next()
        assert first["event"] == "status" and "id" not in first  # status on open, carries no cursor
        await r.until(lambda f: f.get("event") == "progress")
        assert clock.now >= 2.0  # nothing is delivered before it settled
        await r.next(), await r.next()
        assert [p["event"]["event_id"] for p in r.progress()] == ["ev1", "ev2", "ev3"]  # merged by seq
        assert [p["source"] for p in r.progress()] == ["task:t1", "cycle", "task:t1"]
        assert all(set(p["event"]) == {"event_id", "type", "status", "message", "task_id", "node_id", "at"}
                   for p in r.progress())
        assert "ANSWER-KEY" not in json.dumps(r.frames)  # nothing from `data`
        last = [f for f in r.frames if f.get("event") == "progress"][-1]
        assert StreamCursor.decode(last["id"]) == {"task:t1": 3, "cycle": 2}
        reader.stopped = True
        reader.status = TaskStatus.COMPLETED
        end = await r.until(lambda f: f.get("event") == "end")
        assert StreamCursor.decode(end["id"]) == {"task:t1": 3, "cycle": 2}
        assert json.loads(end["data"])["status"] == "COMPLETED"
        assert await r.next() is None and r.closed
        statuses = [f for f in r.frames if f.get("event") == "status"]
        assert len(statuses) == 2  # on open and on the change, not on every poll

    run(go())


def test_end_waits_for_events_written_before_the_stop_state_to_settle() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        reader = FakeReader({"task:t1": "t1"})
        r = Run(service(events, clock), reader)
        await r.next()
        events.insert("t1")
        reader.stopped = True
        reader.status = TaskStatus.COMPLETED
        await r.until(lambda f: f.get("event") == "end")
        assert [p["event"]["event_id"] for p in r.progress()] == ["ev1"]

    run(go())


def test_a_late_commit_with_a_lower_seq_is_still_delivered_within_the_bound() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        reader = FakeReader({"task:t1": "t1"})
        r = Run(service(events, clock), reader)
        await r.next()
        held = events.insert("t1")  # seq 1 allocated, not committed yet
        events.insert("t1")  # seq 2 committed first
        held.visible_at = 1.5  # committed 1.5 s later: within the 2 s bound
        await r.until(lambda f: f.get("event") == "progress")
        await r.next()
        assert [p["event"]["event_id"] for p in r.progress()] == ["ev1", "ev2"]

    run(go())


def test_a_commit_delayed_beyond_the_bound_can_be_passed_the_documented_limit() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        r = Run(service(events, clock), FakeReader({"task:t1": "t1"}))
        await r.next()
        held = events.insert("t1")
        held.visible_at = 30.0  # a stalled transaction: far beyond the 2 s bound
        events.insert("t1")
        await r.until(lambda f: f.get("event") == "progress")
        while clock.now < 40:
            await r.next()
        assert [p["event"]["event_id"] for p in r.progress()] == ["ev2"]  # ev1 was passed: the assumption's edge

    run(go())


def test_a_reconnect_resumes_after_its_cursor_and_replays_an_interrupted_delivery() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        for key in ("c1", "t1", "c1", "t1"):
            events.insert(key)
        reader = FakeReader({"cycle": "c1", "task:t1": "t1"})
        r = Run(service(events, clock), reader)
        await r.until(lambda f: f.get("event") == "progress")
        await r.next()
        received = [f for f in r.frames if f.get("event") == "progress"]
        # the connection drops after the second frame was written but before the client processed it: the client
        # still holds the first frame's cursor
        cursor = StreamCursor.decode(received[0]["id"])
        await r.gen.aclose()
        events.insert("c1")
        again = Run(service(events, clock), reader, cursor)
        await again.until(lambda f: f.get("event") == "progress")
        for _ in range(3):
            await again.next()
        ids = [p["event"]["event_id"] for p in again.progress()]
        assert ids == ["ev2", "ev3", "ev4", "ev5"]  # ev2 replayed: permitted (at least once)
        shown = {p["event"]["event_id"] for p in r.progress()[:1] + again.progress()}  # the client's dedupe set
        assert shown == {f"ev{i}" for i in range(1, 6)}

    run(go())


def test_a_new_source_absent_from_the_cursor_is_read_from_its_start_and_foreign_sources_are_not_read() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        events.insert("c1")
        events.insert("foreign")
        events.insert("t2")
        reader = FakeReader({"cycle": "c1", "task:t2": "t2"})
        cursor = {"cycle": 1, "task:foreign": 0}  # issued before t2 existed; names a source this stream does not own
        r = Run(service(events, clock), reader, cursor)
        await r.until(lambda f: f.get("event") == "progress")
        assert [p["event"]["event_id"] for p in r.progress()] == ["ev3"]
        assert all(asked <= {"c1", "t2"} for asked in events.asked)
        reader.stopped = True
        end = await r.until(lambda f: f.get("event") == "end")
        assert set(StreamCursor.decode(end["id"])) == {"cycle", "task:t2"}

    run(go())


@pytest.mark.parametrize("how", ["access", "read"])
def test_access_loss_or_a_failed_poll_stops_the_stream_without_any_frame(how: str) -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        reader = FakeReader({"task:t1": "t1"})
        allowed = {"ok": True}

        def check() -> None:
            if not allowed["ok"]:
                raise LookupError("not found")

        r = Run(service(events, clock), reader, access=check)
        await r.next()
        events.insert("t1")
        reader.stopped = True  # even at a stop state: no end
        reader.status = TaskStatus.COMPLETED
        if how == "access":
            allowed["ok"] = False
        else:
            reader.fail = RuntimeError("boom with a secret")
        assert await r.next() is None and r.closed
        assert [f.get("event") for f in r.frames] == ["status"]

    run(go())


def test_heartbeat_and_maximum_lifetime() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        events.insert("t1")
        reader = FakeReader({"task:t1": "t1"})
        r = Run(service(events, clock, heartbeat_seconds=15, max_seconds=60), reader)
        await r.until(lambda f: f.get("event") == "progress")
        written = clock.now
        hb = await r.next()
        assert hb == {"comment": True} and 15 <= clock.now - written < 16
        last = await r.until(lambda f: f.get("event") == "reconnect")
        assert 60 <= clock.now < 61 and StreamCursor.decode(last["id"]) == {"task:t1": 1}
        assert await r.next() is None
        beats = sum(1 for f in r.frames if f.get("comment"))
        assert beats >= 3

    run(go())


def test_the_slot_is_freed_when_the_stream_ends_or_is_closed() -> None:
    async def go() -> None:
        clock = Clock()
        events = FakeEvents(clock)
        svc = service(events, clock, max_per_principal=1)
        reader = FakeReader({"task:t1": "t1"})
        reader.stopped = True
        r = Run(svc, reader)
        with pytest.raises(TooManyStreams):
            svc.slots.acquire("p")
        await r.until(lambda f: f.get("event") == "end")
        assert await r.next() is None
        assert svc.slots.open_count() == 0
        r2 = Run(svc, FakeReader({"task:t1": "t1"}))
        await r2.next()
        await r2.gen.aclose()  # the client went away
        assert svc.slots.open_count() == 0

    run(go())
