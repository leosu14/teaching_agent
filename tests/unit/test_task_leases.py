"""Task leases, fencing and versioned writes at the storage boundary (PR18).

Each test drives the repositories directly with a controllable clock: no timing, no sleeps. A "worker" is just the
`Lease` a claim returned; a stale worker is one whose lease expired or was claimed again since.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.artifacts.service import ArtifactService
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.schemas.artifact import ArtifactType
from app.schemas.common import utcnow
from app.schemas.events import Event
from app.schemas.lease import LeaseLost, WriteConflict
from app.schemas.task import Task, TaskStatus
from app.storage.db import create_db
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository, SqlEventRepository, SqlTaskRepository

TTL = timedelta(minutes=5)


class Clock:
    def __init__(self) -> None:
        self.offset = timedelta(0)

    def __call__(self):
        return utcnow() + self.offset


@pytest.fixture
def store(tmp_path):
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    clock = Clock()
    tasks = SqlTaskRepository(sessions, clock)
    task = Task(task_id="task_1", user_id="u", learner_id="l", request="r")
    assert tasks.create(task)
    return tasks, SqlEventRepository(sessions, clock), SqlArtifactRepository(sessions, clock), clock, tmp_path


def test_a_claim_raises_the_fence_and_excludes_others_until_expiry(store) -> None:
    tasks, _, _, clock, _ = store
    first = tasks.claim("task_1", "w1", TTL)
    assert first is not None and first.fence == 1
    assert tasks.claim("task_1", "w2", TTL) is None  # live lease: nobody else executes the task
    clock.offset = TTL + timedelta(seconds=1)
    second = tasks.claim("task_1", "w2", TTL)
    assert second is not None and second.fence == 2  # takeover after expiry invalidates the first holder


def test_a_stale_worker_cannot_write_after_takeover(store) -> None:
    tasks, _, _, clock, _ = store
    old = tasks.claim("task_1", "old", TTL)
    stale_copy = tasks.get("task_1")
    clock.offset = TTL + timedelta(seconds=1)
    new = tasks.claim("task_1", "new", TTL)
    task = tasks.get("task_1")
    task.status = TaskStatus.PLANNING
    tasks.save(task, lease=new)  # the current holder writes
    committed = tasks.get("task_1")

    stale_copy.status = TaskStatus.FAILED
    with pytest.raises(LeaseLost) as raised:
        tasks.save(stale_copy, lease=old)
    assert str(raised.value) == LeaseLost.MESSAGE  # fixed text, no fence or owner in it
    assert tasks.get("task_1") == committed  # the newer state is untouched
    assert tasks.release(old) is False  # nor can the stale worker release the new holder's lease
    assert tasks.lease("task_1").owner == "new"


def test_expiry_alone_stops_a_worker_from_writing(store) -> None:
    tasks, _, _, clock, _ = store
    lease = tasks.claim("task_1", "w", TTL)
    task = tasks.get("task_1")
    clock.offset = TTL + timedelta(seconds=1)  # nobody took over, but the lease ran out
    with pytest.raises(LeaseLost):
        tasks.save(task, lease=lease)
    assert tasks.renew(lease, TTL) is None  # an expired lease is never renewed, only claimed again
    assert tasks.get("task_1").version == 1


def test_renewal_keeps_a_live_worker_current(store) -> None:
    tasks, _, _, clock, _ = store
    lease = tasks.claim("task_1", "w", TTL)
    clock.offset = TTL * 0.9
    renewed = tasks.renew(lease, TTL)
    assert renewed is not None and renewed.fence == lease.fence
    clock.offset = TTL * 1.5  # past the original expiry, within the renewed one
    assert tasks.claim("task_1", "other", TTL) is None
    task = tasks.get("task_1")
    tasks.save(task, lease=lease)
    assert tasks.get("task_1").version == 2


def test_an_unleased_write_from_an_old_version_is_refused(store) -> None:
    tasks, _, _, _, _ = store
    a, b = tasks.get("task_1"), tasks.get("task_1")
    a.metadata["writer"] = "a"
    tasks.save(a)
    assert a.version == 2
    b.metadata["writer"] = "b"
    with pytest.raises(WriteConflict):
        tasks.save(b)  # based on version 1: never silently overwrites version 2
    stored = tasks.get("task_1")
    assert stored.metadata == {"writer": "a"} and stored.version == 2
    reloaded = tasks.get("task_1")  # reload and retry is safe
    reloaded.metadata["writer"] = "b"
    tasks.save(reloaded)
    assert tasks.get("task_1").version == 3


def test_an_unleased_write_waits_for_the_executing_worker(store) -> None:
    tasks, _, _, _, _ = store
    task = tasks.get("task_1")
    lease = tasks.claim("task_1", "w", TTL)
    with pytest.raises(WriteConflict):
        tasks.save(task)
    tasks.release(lease)
    tasks.save(task)


def test_pause_and_cancel_requests_survive_the_holders_writes(store) -> None:
    tasks, _, _, _, _ = store
    lease = tasks.claim("task_1", "w", TTL)
    task = tasks.get("task_1")  # the holder's copy: no request yet
    tasks.request_control("task_1", cancel=True)
    tasks.save(task, lease=lease)  # a checkpoint after the request
    assert tasks.control("task_1").cancel_requested
    tasks.request_control("task_1", pause=True)
    tasks.save(task, lease=lease, clear_control=("pause",))  # the holder acted on the pause
    control = tasks.control("task_1")
    assert control.cancel_requested and not control.pause_requested


def test_create_is_idempotent(store) -> None:
    tasks, _, _, _, _ = store
    assert tasks.create(Task(task_id="task_1", user_id="x", learner_id="y", request="other")) is False
    assert tasks.get("task_1").request == "r"


def test_a_stale_worker_stores_no_event(store) -> None:
    tasks, events, _, clock, _ = store
    old = tasks.claim("task_1", "old", TTL)
    events.append(Event(event_id="evt_a", type="node.started", task_id="task_1"), fence=old)
    clock.offset = TTL + timedelta(seconds=1)
    tasks.claim("task_1", "new", TTL)
    with pytest.raises(LeaseLost):
        events.append(Event(event_id="evt_b", type="task.completed", task_id="task_1"), fence=old)

    bus = EventBus()  # through the bus (as workers emit): refused quietly, never breaking the caller
    bus.subscribe_fenced(events.append)
    bus.emit("task.completed", task_id="task_1", fence=old)
    assert [e.event_id for e in events.list_for_task("task_1")] == ["evt_a"]


def test_a_stale_worker_stores_no_artifact_and_overwrites_no_object(store) -> None:
    tasks, events, artifact_repo, clock, tmp_path = store
    service = ArtifactService(artifact_repo, FilesystemObjectStore(tmp_path / "objects"))
    bus = EventBus()
    old = tasks.claim("task_1", "old", TTL)
    clock.offset = TTL + timedelta(seconds=1)
    new = tasks.claim("task_1", "new", TTL)

    def store_as(lease, text: str):
        return service.store(task_id="task_1", name="lesson", type=ArtifactType.LESSON, media_type="text/plain",
                             content=text.encode(), provider="test", parent_ids=[], metadata={},
                             scope=ExecutionScope(events=bus, usage=UsageLedger(), task_id="task_1", fence=lease))

    kept = store_as(new, "the new worker's lesson")
    with pytest.raises(LeaseLost):
        store_as(old, "the stale worker's lesson")
    stored = service.list_for_task("task_1")
    assert [a.artifact_id for a in stored] == [kept.artifact_id] and kept.version == 1
    assert service.verify(kept) is None  # its object was not overwritten by the stale worker's put
    with pytest.raises(WriteConflict):  # the same (task, name, version) is recorded once
        artifact_repo.add(kept.model_copy(update={"artifact_id": "art_dup"}))


def test_a_stale_workers_orphan_object_never_becomes_the_recorded_one(store) -> None:
    tasks, _, artifact_repo, clock, tmp_path = store
    service = ArtifactService(artifact_repo, FilesystemObjectStore(tmp_path / "objects"))
    old = tasks.claim("task_1", "old", TTL)
    clock.offset = TTL + timedelta(seconds=1)
    new = tasks.claim("task_1", "new", TTL)

    def store_as(lease, text: str):
        return service.store(task_id="task_1", name="lesson", type=ArtifactType.LESSON, media_type="text/plain",
                             content=text.encode(), provider="test", parent_ids=[], metadata={},
                             scope=ExecutionScope(events=EventBus(), usage=UsageLedger(), task_id="task_1",
                                                  fence=lease))

    with pytest.raises(LeaseLost):  # the stale worker uploads first; its row is refused
        store_as(old, "the stale worker's lesson")
    kept = store_as(new, "the new worker's lesson")
    assert kept.version == 1 and service.read(kept.artifact_id) == b"the new worker's lesson"
    assert service.verify(kept) is None


def test_a_database_from_before_pr18_gets_the_lease_columns(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:  # the PR17 schema of the two changed tables, with a stored task
        db.execute("CREATE TABLE tasks (task_id VARCHAR(64) PRIMARY KEY, user_id VARCHAR(128), learner_id "
                   "VARCHAR(128), status VARCHAR(32), created_at DATETIME, updated_at DATETIME, body TEXT)")
        db.execute("CREATE TABLE learning_cycles (cycle_id VARCHAR(64) PRIMARY KEY, learner_id VARCHAR(128), "
                   "status VARCHAR(32), version INTEGER, created_at DATETIME, body TEXT)")
        task = Task(task_id="task_old", user_id="u", learner_id="l", request="r")
        db.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (task.task_id, "u", "l", "CREATED", "2026-01-01 00:00:00", "2026-01-01 00:00:00",
                    task.model_dump_json()))
    tasks = SqlTaskRepository(create_db(f"sqlite:///{path}"))
    stored = tasks.get("task_old")
    assert stored.version == 1 and not stored.control.cancel_requested
    lease = tasks.claim("task_old", "w", TTL)
    assert lease is not None and lease.fence == 1
    tasks.save(stored, lease=lease)
    assert tasks.get("task_old").version == 2
