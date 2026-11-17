"""Progress streams: a task's or a learning cycle's progress as server-sent events (issue #54).

Read-only. A stream polls the database on the server (there is no pub/sub: the worker is another process) and
writes `text/event-stream` frames built only from the learner-safe views the GET routes already serve: `TaskView`,
`LearningCycleView` and `LearnerEvent`. Nothing is copied from an event's `data`.

Frames:
- `status`: the task's or cycle's view, on open and whenever it changes;
- `progress`: one learner event and its source, `{"source": "task:<id>" | "cycle", "event": LearnerEvent}`;
- `end`: only at a stop state (`task_stopped`, `cycle_stopped`), with the final view;
- `reconnect`: at the maximum lifetime; the client reconnects (and is authenticated again);
- `: keepalive` comments, so an idle stream outlives proxy read timeouts.

Every `progress`, `end` and `reconnect` frame carries the cursor as its SSE `id` (`StreamCursor`): the position
(`events.seq`) delivered so far from each source. A reconnect sends it back as `Last-Event-ID`.

Delivery is at least once. Within one connection a source's position only increases, so nothing is sent twice; a
reconnect resumes strictly after the positions in the cursor the client last received, so an event whose frame was
written but whose cursor never arrived is sent again, and the client deduplicates by `event_id`.

Settled reads. `seq` is allocated at insert but visible at commit, so on PostgreSQL a lower `seq` can become visible
after a higher one. Each poll records the table's highest visible `seq`; events are delivered only up to the highest
value recorded at least `settle_seconds` earlier. That is an engineering assumption (each event append is its own
short transaction), not a guarantee: a commit delayed beyond the bound can still be passed (docs/architecture.md).

Access loss. Each poll re-runs the caller's access rule (the GET route's) after its reads and before anything read
in it is sent. When the rule fails, or anything in the poll raises, the stream stops at once: no further frame, not
even `end`. The data read in that poll is discarded.

The loop is a coroutine: it sleeps on the event loop between polls and does each poll's reads in a worker thread,
so an idle stream holds no thread. Open streams are capped per principal and per process (`TooManyStreams`)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from app.schemas.events import LearnerEvent
from app.schemas.job import ACTIVE_JOB_STATUSES, TaskJob
from app.schemas.learning_cycle import (
    TERMINAL_CYCLE_STATUSES,
    CycleStatus,
    LearningCycle,
    LearningCycleView,
    StepKind,
)
from app.schemas.task import Task, TaskStatus, TaskView
from app.services.learning_cycles import LEARNER_WAITS, LearningCycleService
from app.services.tasks import TaskService
from app.storage.repositories import SqlEventRepository

logger = logging.getLogger(__name__)

CYCLE_SOURCE = "cycle"
TASK_PREFIX = "task:"
CURSOR_VERSION = "v1"
MAX_CURSOR_LENGTH = 4096
MAX_CURSOR_SOURCES = 64
_SOURCE = re.compile(r"^(cycle|task:[A-Za-z0-9_.:\-]{1,128})$")
TASK_WORKING = frozenset({TaskStatus.CREATED, TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.REVIEWING})
TASK_STOPPED = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.PAUSED})
TASK_STEP_KINDS = frozenset({StepKind.LESSON, StepKind.EVALUATION})  # steps whose child is a task with events
EVENTS_PER_POLL = 500
# The polls' reads run on their own small pool, not the event loop's default one (shared with the rest of the API):
# many open streams wait their turn there instead of starving other work.
STREAM_THREADS = 8


class InvalidCursor(ValueError):
    """A `Last-Event-ID` this server cannot parse (400)."""


class TooManyStreams(Exception):
    """The principal, or this process, has as many open streams as allowed (429)."""


# --- the cursor ----------------------------------------------------------------------------------------------------


class StreamCursor:
    """`v1.` + base64url(JSON {source: position}). Not signed: it holds only positions in the caller's own streams,
    and a stream reads only the sources it owns, whatever a cursor names."""

    @staticmethod
    def encode(positions: dict[str, int]) -> str:
        raw = json.dumps(positions, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{CURSOR_VERSION}.{base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')}"

    @staticmethod
    def decode(value: str | None) -> dict[str, int]:
        if value is None or not value.strip():
            return {}
        value = value.strip()
        if len(value) > MAX_CURSOR_LENGTH:
            raise InvalidCursor("the cursor is too long")
        version, _, body = value.partition(".")
        if version != CURSOR_VERSION or not body:
            raise InvalidCursor("unknown cursor version")
        try:
            raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
            data = json.loads(raw.decode("utf-8"))
        except (binascii.Error, ValueError, UnicodeDecodeError):
            raise InvalidCursor("the cursor cannot be read") from None
        if not isinstance(data, dict) or len(data) > MAX_CURSOR_SOURCES:
            raise InvalidCursor("the cursor cannot be read")
        positions: dict[str, int] = {}
        for source, position in data.items():
            if not isinstance(source, str) or not _SOURCE.match(source) or isinstance(position, bool) \
                    or not isinstance(position, int) or position < 0:
                raise InvalidCursor("the cursor cannot be read")
            positions[source] = position
        return positions


def task_source(task_id: str) -> str:
    return f"{TASK_PREFIX}{task_id}"


# --- stop states -----------------------------------------------------------------------------------------------------


def task_owes_work(task: Task) -> bool:
    """CREATED, PLANNING, RUNNING or REVIEWING, or WAITING on something other than the learner (as `_observe`)."""
    return task.status in TASK_WORKING or (task.status == TaskStatus.WAITING
                                           and (task.waiting is None or task.waiting.kind not in LEARNER_WAITS))


def job_active(job: TaskJob | None) -> bool:
    return job is not None and job.status in ACTIVE_JOB_STATUSES


def task_stopped(task: Task, job: TaskJob | None) -> bool:
    """No job QUEUED or RUNNING, and COMPLETED, FAILED, CANCELLED, PAUSED or WAITING for the learner."""
    if job_active(job):
        return False
    return task.status in TASK_STOPPED or (task.status == TaskStatus.WAITING and task.waiting is not None
                                           and task.waiting.kind in LEARNER_WAITS)


def cycle_stopped(view: LearningCycleView, *, lease_live: bool, child_busy: bool, after_version: int | None) -> bool:
    """Not in progress (not RUNNING, no live drive lease, the current child task owes no work and has no active
    job), and terminal, or WAITING or BLOCKED at a version above `after_version` (any version without one)."""
    if view.status == CycleStatus.RUNNING or lease_live or child_busy:
        return False
    if view.status in TERMINAL_CYCLE_STATUSES:
        return True
    if view.status in (CycleStatus.WAITING, CycleStatus.BLOCKED):
        return after_version is None or view.version > after_version
    return False


# --- limits ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamLimits:
    max_seconds: float = 600.0
    poll_seconds: float = 1.0
    settle_seconds: float = 2.0
    heartbeat_seconds: float = 15.0
    max_per_principal: int = 4
    max_streams: int = 200


class StreamSlots:
    """Counts open streams per principal and in total (this process)."""

    def __init__(self, *, per_principal: int, total: int) -> None:
        self._per_principal = per_principal
        self._total = total
        self._lock = threading.Lock()
        self._open: dict[str, int] = {}

    def acquire(self, principal: str) -> StreamSlot:
        with self._lock:
            if sum(self._open.values()) >= self._total or self._open.get(principal, 0) >= self._per_principal:
                raise TooManyStreams("too many open progress streams: close one or retry later")
            self._open[principal] = self._open.get(principal, 0) + 1
        return StreamSlot(self, principal)

    def _release(self, principal: str) -> None:
        with self._lock:
            left = self._open.get(principal, 0) - 1
            if left > 0:
                self._open[principal] = left
            else:
                self._open.pop(principal, None)

    def open_count(self, principal: str | None = None) -> int:
        with self._lock:
            return sum(self._open.values()) if principal is None else self._open.get(principal, 0)


class StreamSlot:
    def __init__(self, slots: StreamSlots, principal: str) -> None:
        self._slots = slots
        self._principal = principal
        self._held = True
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if not self._held:
                return
            self._held = False
        self._slots._release(self._principal)


# --- frames ----------------------------------------------------------------------------------------------------------


def frame(event: str, data: str, cursor: str | None = None) -> str:
    head = f"id: {cursor}\n" if cursor is not None else ""
    return f"{head}event: {event}\ndata: {data}\n\n"


HEARTBEAT = ": keepalive\n\n"


@dataclass
class Snapshot:
    """One poll's reads: the view, whether it is a stop state, the owned sources (name -> stored key) and the
    settled events read from them."""

    view: TaskView | LearningCycleView
    stopped: bool
    sources: dict[str, str]
    top: int = 0
    settled: int | None = None
    events: list[tuple[int, str, LearnerEvent]] = field(default_factory=list)
    truncated: bool = False


class _Settle:
    """The settled bound: the highest `seq` read at least `seconds` before now."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._reads: deque[tuple[float, int]] = deque()

    def record(self, at: float, top: int) -> int | None:
        self._reads.append((at, top))
        horizon = at - self._seconds
        while len(self._reads) > 1 and self._reads[1][0] <= horizon:
            self._reads.popleft()
        first_at, first_top = self._reads[0]
        return first_top if first_at <= horizon else None


# --- the service ----------------------------------------------------------------------------------------------------


AccessCheck = Callable[[], object]  # raises when the caller may no longer read the resource


class ProgressStream:
    """One open stream: iterate its frames; `aclose()` (always called by the API's response) frees its slot."""

    def __init__(self, frames: AsyncIterator[str], slot: StreamSlot) -> None:
        self._frames = frames
        self._slot = slot

    def __aiter__(self) -> AsyncIterator[str]:
        return self._frames

    async def aclose(self) -> None:
        try:
            await self._frames.aclose()  # type: ignore[attr-defined]
        finally:
            self._slot.release()


class ProgressStreamService:
    def __init__(self, *, tasks: TaskService, cycles: LearningCycleService, events: SqlEventRepository,
                 limits: StreamLimits = StreamLimits(), clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[object]] = asyncio.sleep) -> None:
        self._tasks = tasks
        self._cycles = cycles
        self._events = events
        self.limits = limits
        self._clock = clock
        self._sleep = sleep
        self.slots = StreamSlots(per_principal=limits.max_per_principal, total=limits.max_streams)
        self._pool = ThreadPoolExecutor(max_workers=STREAM_THREADS, thread_name_prefix="progress-stream")

    def close(self) -> None:
        """Let the poll threads go (the container is closing): a poll in progress finishes, no new one starts."""
        self._pool.shutdown(wait=False)

    # --- opening -------------------------------------------------------------------------------------------------

    def task(self, task_id: str, *, principal: str, check_access: AccessCheck,
             cursor: str | None = None) -> ProgressStream:
        """The task's stream. The caller has checked access already (the route's dependency); `check_access` is
        the same rule, re-run before every send. InvalidCursor, TooManyStreams before anything is read."""
        positions = StreamCursor.decode(cursor)
        slot = self.slots.acquire(principal)
        reader = _TaskReader(self._tasks, task_id)
        return ProgressStream(self._run(reader, positions, check_access, slot), slot)

    def cycle(self, cycle_id: str, *, principal: str, check_access: AccessCheck, cursor: str | None = None,
              after_version: int | None = None) -> ProgressStream:
        """The cycle's stream: its own events and its child tasks'. `after_version` (the version of the view the
        client showed when it sent a POST) only decides whether WAITING or BLOCKED is new enough to end it."""
        positions = StreamCursor.decode(cursor)
        slot = self.slots.acquire(principal)
        reader = _CycleReader(self._cycles, self._tasks, cycle_id, after_version)
        return ProgressStream(self._run(reader, positions, check_access, slot), slot)

    # --- the loop ------------------------------------------------------------------------------------------------

    def _poll(self, reader: _Reader, positions: dict[str, int], settle: _Settle,
              check_access: AccessCheck) -> Snapshot:
        """One poll, in a worker thread: the state, then the table's highest `seq`, then the settled events of the
        owned sources, then the access rule (so nothing read here is sent unless access still holds after it)."""
        snap = reader.read()
        snap.top = self._events.max_seq()  # after the state: every event written before the state was read is <= top
        snap.settled = settle.record(self._clock(), snap.top)
        if snap.settled is not None:
            wanted = {key: positions.get(name, 0) for name, key in snap.sources.items()}
            by_key = {key: name for name, key in snap.sources.items()}
            rows = self._events.after(wanted, snap.settled, limit=EVENTS_PER_POLL)
            snap.truncated = len(rows) >= EVENTS_PER_POLL
            snap.events = [(seq, by_key[e.task_id], LearnerEvent.of(e)) for seq, e in rows
                           if e.task_id in by_key]
        check_access()
        return snap

    async def _run(self, reader: _Reader, start: dict[str, int], check_access: AccessCheck,
                   slot: StreamSlot) -> AsyncIterator[str]:
        limits = self.limits
        settle = _Settle(limits.settle_seconds)
        positions = dict(start)  # filtered to the owned sources at every poll: a foreign source is never read
        opened = last_write = self._clock()
        last_status: str | None = None
        ending_at: int | None = None  # the table's top when the stop state was first seen: delivered before `end`
        try:
            while True:
                try:
                    snap = await asyncio.get_running_loop().run_in_executor(
                        self._pool, self._poll, reader, positions, settle, check_access)
                except Exception as exc:  # noqa: BLE001 - access lost or a failed read: stop without a frame
                    reader.lost(exc)
                    return
                positions = {name: positions.get(name, 0) for name in snap.sources}
                out: list[str] = []
                status = snap.view.model_dump_json()
                if status != last_status:
                    out.append(frame("status", status))
                    last_status = status
                for seq, name, event in snap.events:
                    positions[name] = max(positions[name], seq)
                    out.append(frame("progress", json.dumps({"source": name, "event": event.model_dump(mode="json")},
                                                            separators=(",", ":")),
                                     StreamCursor.encode(positions)))
                done = False
                if snap.stopped:
                    ending_at = snap.top if ending_at is None else ending_at
                    if snap.settled is not None and snap.settled >= ending_at and not snap.truncated:
                        out.append(frame("end", status, StreamCursor.encode(positions)))
                        done = True
                else:
                    ending_at = None
                now = self._clock()
                if not done and now - opened >= limits.max_seconds:
                    out.append(frame("reconnect", "{}", StreamCursor.encode(positions)))
                    done = True
                if not out and now - last_write >= limits.heartbeat_seconds:
                    out.append(HEARTBEAT)
                if out:
                    for chunk in out:
                        yield chunk
                    last_write = now
                if done:
                    return
                await self._sleep(0 if snap.truncated else limits.poll_seconds)
        finally:
            slot.release()


# --- readers ---------------------------------------------------------------------------------------------------------


class _Reader:
    def read(self) -> Snapshot:  # pragma: no cover - interface
        raise NotImplementedError

    def lost(self, exc: Exception) -> None:
        # The type only: the message could name the resource, or quote stored text.
        logger.info("progress stream closed: %s", type(exc).__name__)


class _TaskReader(_Reader):
    def __init__(self, tasks: TaskService, task_id: str) -> None:
        self._tasks = tasks
        self._task_id = task_id

    def read(self) -> Snapshot:
        task = self._tasks.get(self._task_id)
        job = self._tasks.job(self._task_id)
        return Snapshot(view=TaskView.of(task, job), stopped=task_stopped(task, job),
                        sources={task_source(task.task_id): task.task_id})


class _CycleReader(_Reader):
    def __init__(self, cycles: LearningCycleService, tasks: TaskService, cycle_id: str,
                 after_version: int | None) -> None:
        self._cycles = cycles
        self._tasks = tasks
        self._cycle_id = cycle_id
        self._after_version = after_version
        self._children: dict[str, bool] = {}  # child task id -> it belongs to the cycle's learner (checked once)

    def read(self) -> Snapshot:
        # The same read as GET /learning-cycles/{id}: it folds in what the children did and never drives one.
        record, view = asyncio.run(self._cycles.read(self._cycle_id))
        sources = {CYCLE_SOURCE: record.cycle_id}
        for step in record.steps:
            if step.kind in TASK_STEP_KINDS and step.child_id and self._own(record, step.child_id):
                sources[task_source(step.child_id)] = step.child_id
        return Snapshot(view=view, stopped=cycle_stopped(view, lease_live=self._cycles.lease_live(record),
                                                         child_busy=self._child_busy(record),
                                                         after_version=self._after_version),
                        sources=sources)

    def _own(self, record: LearningCycle, child_id: str) -> bool:
        if child_id not in self._children:
            self._children[child_id] = self._tasks.get(child_id).learner_id == record.learner_id
        return self._children[child_id]

    def _child_busy(self, record: LearningCycle) -> bool:
        step = record.current_step
        if step is None or step.kind not in TASK_STEP_KINDS or step.child_id is None:
            return False
        task = self._tasks.get(step.child_id)
        return task_owes_work(task) or job_active(self._tasks.job(step.child_id))
