"""Learner data deletion on PostgreSQL (MVP item 9). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to.

The database is shared by every replica and by other learners, so the deletion must find exactly one learner's rows
there: a learner whose id holds LIKE wildcards (`_`, `%`) must not take another learner's events with them. A worker
holding a live lease blocks the deletion; a worker whose lease ran out cannot bring anything back once the rows are
gone, because its writes are fenced by the task row. Concurrent deletions of one learner, sent to two replicas, all
succeed and leave nothing. Every run uses fresh ids, so the database can be reused."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.common import new_id
from app.schemas.events import Event
from app.schemas.lease import LeaseLost
from app.schemas.learner import LearnerProfileInput
from app.services.container import build_container
from app.storage.orm import EventRow, SpendCounterRow, TaskRow
from scripts.run_learning_cycle_demo import scenario, settings
from tests.auth import TestAuth, client_for

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_curriculum_demo as curriculum_demo  # noqa: E402

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL deletion tests")


def replica(data_dir: Path):
    return build_container(settings(data_dir, database_url=URL, log_json=False),
                           llm_providers={"mock": MockLLMProvider(default_responders())})


def placed_learner(c, learner_id: str) -> str:
    fx = curriculum_demo._fixture("learner.json")
    c.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(fx["profile"]))

    async def place() -> None:
        for domain, evidence in curriculum_demo.placement_evidence(learner_id, fx["placement"]).items():
            await c.learner_service.record_evidence(learner_id, domain, evidence)
        for concept in scenario()["secured_concepts"]:
            await c.learner_service.record_evidence(
                learner_id, "spanish", curriculum_demo.exercise_evidence(learner_id, concept, 4, "secured"))

    asyncio.run(place())
    return learner_id


def start_learning(client, learner_id: str) -> dict:
    """A goal, its curriculum and a started learning cycle: a lesson generated and stored, the cycle WAITING."""
    goal = client.post(f"/learners/{learner_id}/goals", json={
        "title": "Reach B1 Spanish", "domain": "spanish", "target_level": "B1", "idempotency_key": new_id("goal"),
        "metadata": {"framework_id": "cefr"}})
    assert goal.status_code == 201, goal.text
    assert client.post(f"/goals/{goal.json()['goal_id']}/curriculum").status_code == 201
    view = client.post(f"/learners/{learner_id}/learning-cycles", json={"idempotency_key": new_id("web")})
    assert view.status_code in (200, 201), view.text
    return view.json()


def events_naming(c, learner_id: str) -> int:
    with c._sessions() as s:
        return s.scalar(select(func.count()).select_from(EventRow).where(
            EventRow.body.contains(f'"learner_id":"{learner_id}"', autoescape=True)))


def test_a_deletion_on_a_shared_database_takes_one_learners_data_only(tmp_path) -> None:
    first, second = replica(tmp_path / "data"), replica(tmp_path / "data")
    try:
        run = new_id("r").replace("_", "")
        doomed = placed_learner(first, f"del_%{run}")  # LIKE wildcards in the id
        lookalike = placed_learner(first, f"delx%{run}")  # what `_` would match
        other = placed_learner(first, f"delxx{run}")  # what `%` would match
        auth = TestAuth()
        with client_for(first, doomed, auth=auth) as a, client_for(second, doomed, auth=auth) as a2, \
                client_for(first, lookalike, other, auth=auth) as b:
            started = start_learning(a, doomed)
            start_learning(b, lookalike)
            start_learning(b, other)
            assert events_naming(first, doomed) and events_naming(first, lookalike) and events_naming(first, other)
            repo = first.learner_deletion.repository
            objects = repo.objects(doomed).delete
            assert objects
            with first._sessions() as s:
                counters = s.scalar(select(func.count()).select_from(SpendCounterRow))
            b_before = {lid: b.get(f"/learners/{lid}").json() for lid in (lookalike, other)}

            r = a2.delete(f"/learners/{doomed}")  # on the other replica
            assert r.status_code == 200, r.text
            assert r.json()["records"]["tasks"] >= 1 and r.json()["records"]["learning_cycles"] == 1
            assert not repo.holds_data(doomed) and events_naming(first, doomed) == 0
            assert a.get(f"/learning-cycles/{started['cycle_id']}").status_code == 404
            for uri in objects:
                assert not first.artifacts.object_exists(uri), uri

            for lid in (lookalike, other):
                assert repo.holds_data(lid) and events_naming(first, lid)
                assert b.get(f"/learners/{lid}").json() == b_before[lid]
            assert b.get(f"/learners/{lookalike}/learning-cycles").json()
            with first._sessions() as s:
                assert s.scalar(select(func.count()).select_from(SpendCounterRow)) == counters  # caps keep counting
    finally:
        first.close()
        second.close()


def test_a_live_lease_blocks_and_an_expired_one_cannot_write_after_the_deletion(tmp_path) -> None:
    c = replica(tmp_path / "data")
    try:
        learner = placed_learner(c, new_id("pg-del"))
        auth = TestAuth()
        with client_for(c, learner, auth=auth) as a:
            start_learning(a, learner)
            with c._sessions() as s:
                task_id = s.scalar(select(TaskRow.task_id).where(TaskRow.learner_id == learner).limit(1))
            tasks = c.task_service._tasks
            live = tasks.claim(task_id, "worker-live", timedelta(minutes=5))
            assert live is not None
            busy = a.delete(f"/learners/{learner}")
            assert busy.status_code == 409 and c.learner_deletion.repository.holds_data(learner)
            assert tasks.release(live)

            stale = tasks.claim(task_id, "worker-stale", timedelta(milliseconds=1))  # runs out at once
            assert stale is not None
            threading.Event().wait(0.05)
            assert a.delete(f"/learners/{learner}").status_code == 200
            event = Event(event_id=new_id("evt"), type="ARTIFACT_CREATED", task_id=task_id,
                          data={"learner_id": learner})
            with pytest.raises(LeaseLost):  # the stale worker's write: its task row is gone
                c.task_service._events.append(event, fence=stale)
            assert not c.learner_deletion.repository.holds_data(learner)
    finally:
        c.close()


def test_concurrent_deletions_of_one_learner_all_succeed(tmp_path) -> None:
    first, second = replica(tmp_path / "data"), replica(tmp_path / "data")
    try:
        learner = placed_learner(first, new_id("pg-del"))
        auth = TestAuth()
        with client_for(first, learner, auth=auth) as a:
            start_learning(a, learner)
        clients = [client_for(c, learner, auth=auth) for c in (first, second, first, second)]
        statuses, errors, barrier = [], [], threading.Barrier(len(clients))

        def delete(client) -> None:
            try:
                barrier.wait()
                statuses.append(client.delete(f"/learners/{learner}").status_code)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=delete, args=(c,)) for c in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for c in clients:
            c.close()
        assert not errors and statuses == [200] * len(clients), (statuses, errors)
        assert not first.learner_deletion.repository.holds_data(learner)
    finally:
        first.close()
        second.close()
