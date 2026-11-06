"""Fenced learner-memory and curriculum writes, versioned profiles and the sweeper on PostgreSQL (MVP item 2). Opt-in
like the other PostgreSQL tests: skipped unless TA_TEST_POSTGRES_URL names a database the test may write to.

SQLite serializes writers, so it cannot show what happens when writers really race. Here row locks decide: many
writers changing one learner's profile at once all keep their change (the version check, then a redo), a stale
worker's write never commits once another worker took its task over (the fence's row lock against the claim), one
read of a curriculum pointer can be saved once, and concurrent sweepers queue one recovery job per stranded task.
Every run uses fresh ids, so the database can be reused.
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta

import pytest

from app.learner.frameworks import default_frameworks
from app.learner.memory import LearnerMemoryService
from app.runtime.orchestrator.orchestrator import TASK_LEASE
from app.schemas.common import new_id, utcnow
from app.schemas.curriculum import CurriculumRecord, RecordConflict
from app.schemas.learner import LearnerProfileInput, LearningEvidence
from app.schemas.lease import LeaseLost, fenced_writes
from app.schemas.task import Task, TaskStatus
from app.services.sweeper import STRANDED, TaskSweeper
from app.services.tasks import TaskService
from app.storage.db import create_db, dispose
from app.storage.repositories import (
    SqlCurriculumRepository,
    SqlEvidenceRepository,
    SqlGoalRepository,
    SqlLearnerRepository,
    SqlLearningEventRepository,
    SqlTaskJobRepository,
    SqlTaskRepository,
)
from tests.integration.test_fenced_learner_writes import taken_over, writes

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL learner fence tests")
THREADS = 8
AHEAD = TASK_LEASE + timedelta(seconds=1)


class Clock:
    def __init__(self, offset: timedelta = timedelta(0)) -> None:
        self.offset = offset

    def __call__(self):
        return utcnow() + self.offset


@pytest.fixture
def sessions():
    factory = create_db(URL)
    yield factory
    dispose(factory)


def memory(sessions) -> LearnerMemoryService:
    """One writer's view of learner memory (its own repositories, as another process would have)."""
    return LearnerMemoryService(SqlLearnerRepository(sessions), default_frameworks(),
                                evidence=SqlEvidenceRepository(sessions), events=SqlLearningEventRepository(sessions),
                                goals=SqlGoalRepository(sessions))


def run_threads(target, n: int = THREADS) -> None:
    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive()


def test_concurrent_changes_to_one_profile_are_all_kept(sessions) -> None:
    lid = new_id("pg-learner")
    memory(sessions).get_or_create(lid)
    barrier = threading.Barrier(THREADS)
    errors: list[BaseException] = []

    def change(i: int) -> None:
        mine = memory(sessions)
        barrier.wait()
        try:
            if i == 0:  # a profile edit racing the evidence writers
                mine.upsert(lid, LearnerProfileInput(display_name="edited"))
                return
            ev = LearningEvidence(evidence_id=LearningEvidence.id_for(lid, "exercise", f"ex{i}", f"c{i}"),
                                  learner_id=lid, concept_id=f"c{i}", source_type="exercise", source_ref=f"ex{i}",
                                  correctness="correct", score=1.0, difficulty=0.5, timestamp=utcnow())
            mine.record_evidence(lid, "spanish", [ev], [])
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    run_threads(change)
    assert errors == []
    profile = memory(sessions).get(lid)
    assert profile.display_name == "edited"
    assert sorted(profile.concepts) == sorted(f"c{i}" for i in range(1, THREADS))  # no change was lost
    assert all(profile.concepts[f"c{i}"].evidence_count == 1 for i in range(1, THREADS))


@pytest.mark.parametrize("name", ["profile", "evidence", "learning_event", "goal", "curriculum_record",
                                  "curriculum_progress"])
def test_a_stale_lease_stores_no_learner_or_curriculum_write(sessions, name) -> None:
    stale, current = taken_over(sessions)
    write, stored = writes(sessions)[name]
    with fenced_writes(stale), pytest.raises(LeaseLost):
        write()
    assert not stored()
    with fenced_writes(current):
        write()
    assert stored()


def test_no_stale_write_commits_once_the_task_is_taken_over(sessions) -> None:
    """A worker keeps writing evidence under its lease while another worker takes the task over. Every write that
    committed did so before the takeover: once one is refused, none after it succeeds, and none started after the
    takeover committed."""
    tasks = SqlTaskRepository(sessions)
    task = Task(task_id=new_id("pgtask"), user_id="u", learner_id="l", request="r")
    tasks.create(task)
    lease = tasks.claim(task.task_id, "old", TASK_LEASE)
    evidence = SqlEvidenceRepository(sessions)
    lid = new_id("pg-learner")
    outcomes: list[tuple[int, bool]] = []
    taken = threading.Event()
    started = threading.Event()
    after_takeover: list[bool] = []

    def write(_: int) -> None:
        with fenced_writes(lease):
            for n in range(100_000):
                was_taken = taken.is_set()
                ev = LearningEvidence(evidence_id=LearningEvidence.id_for(lid, "exercise", f"e{n}", "c"),
                                      learner_id=lid, concept_id="c", source_type="exercise", source_ref=f"e{n}",
                                      correctness="correct", score=1.0, difficulty=0.5, timestamp=utcnow())
                try:
                    evidence.add(ev)
                    ok = True
                except LeaseLost:
                    ok = False
                outcomes.append((n, ok))
                if was_taken:
                    after_takeover.append(ok)
                if n == 20:
                    started.set()
                if not ok and len(after_takeover) > 5:
                    return

    writer = threading.Thread(target=write, args=(0,))
    writer.start()
    assert started.wait(60)
    assert SqlTaskRepository(sessions, Clock(AHEAD)).claim(task.task_id, "new", TASK_LEASE) is not None
    taken.set()
    writer.join(timeout=120)
    results = [ok for _, ok in outcomes]
    first_refused = results.index(False)
    assert all(results[:first_refused]) and not any(results[first_refused:])
    assert after_takeover and not any(after_takeover)
    stored = {e.source_ref for e in evidence.for_learner(lid)}
    assert stored == {f"e{n}" for n, ok in outcomes if ok}


def test_one_read_of_a_curriculum_pointer_is_saved_once(sessions) -> None:
    curricula = SqlCurriculumRepository(sessions)
    now = utcnow()
    cid = new_id("pg-cur")
    curricula.save_record(CurriculumRecord(curriculum_id=cid, learner_id="l", goal_id="g", title="t",
                                           domain="spanish", current_version=1, current_version_id="v1",
                                           created_at=now, updated_at=now))
    read = curricula.record(cid)
    barrier = threading.Barrier(THREADS)
    saved: list[int] = []
    conflicts: list[int] = []

    def move(i: int) -> None:
        barrier.wait()
        try:
            curricula.save_record(read.model_copy(update={"current_version": 2,
                                                          "current_version_id": f"v2-{i}"}))
            saved.append(i)
        except RecordConflict:
            conflicts.append(i)

    run_threads(move)
    assert len(saved) == 1 and len(conflicts) == THREADS - 1
    assert curricula.record(cid).current_version_id == f"v2-{saved[0]}"


def test_concurrent_sweepers_queue_one_recovery_job_per_stranded_task(sessions) -> None:
    clock = Clock(AHEAD)
    tasks = SqlTaskRepository(sessions)
    ids = []
    for _ in range(3):
        task = Task(task_id=new_id("pgtask"), user_id="u", learner_id="l", request="r", status=TaskStatus.RUNNING)
        tasks.create(task)
        assert tasks.claim(task.task_id, "killed", TASK_LEASE) is not None  # its worker dies holding the lease
        ids.append(task.task_id)
    barrier = threading.Barrier(THREADS)
    recovered: list[str] = []
    lock = threading.Lock()

    def sweep(_: int) -> None:
        jobs = SqlTaskJobRepository(sessions, clock)
        service = TaskService(None, tasks, None, None, jobs=jobs)  # only its job queue is used by the sweeper
        sweeper = TaskSweeper(SqlTaskRepository(sessions, clock), jobs, service, batch=100_000, clock=clock)
        barrier.wait()
        report = sweeper.sweep()
        with lock:
            recovered.extend(report.recovered)

    try:
        run_threads(sweep)
        jobs = SqlTaskJobRepository(sessions)
        for task_id in ids:
            assert recovered.count(task_id) == 1
            assert [j.status.value for j in jobs.for_task(task_id)] == ["QUEUED"]
        assert not set(ids) & set(SqlTaskRepository(sessions, clock).stranded(STRANDED,
                                                                                idle_before=clock(), limit=100_000))
    finally:
        jobs = SqlTaskJobRepository(sessions)
        for task_id in set(recovered):  # leave nothing queued for later runs (and other runs' leftovers)
            jobs.cancel_queued(task_id)
