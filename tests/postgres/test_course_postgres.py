"""The bundled course on PostgreSQL (MVP item 5). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to.

An API process and a worker process each load the course at startup and share one database. Here: a B1 goal over the
whole course (every concept a target) and its curriculum are stored and read back by another process identically;
the next action and its lesson run there on the course's probes; and concurrent builds of one goal's curriculum from
several processes store one version. Every run uses fresh ids, so the database can be reused."""

from __future__ import annotations

import asyncio
import os
import threading

import pytest

from app.config.settings import COURSES_DIR, Settings
from app.content.course import read_course
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.runtime.orchestrator.orchestrator import TaskBusy
from app.schemas.common import new_id
from app.schemas.curriculum import GoalInput, LearningActionType, curriculum_id_for
from app.schemas.learner import LearnerProfileInput
from app.schemas.task import TaskStatus
from app.services.container import build_container
from tests.conftest import LEARNER as DEMO_LEARNER
from tests.integration.test_course_flow import answer_all

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL course tests")
COURSE = read_course(COURSES_DIR / "spanish-a1-b1.json")
ALL = {c.concept_id for _, c in COURSE.concepts()}
PROCESSES = 4


def container(data_dir):
    settings = Settings(data_dir=data_dir, database_url=URL, log_json=False, courses="spanish-a1-b1",
                        visual_failure_policy="continue")
    return build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})


def learner(c) -> str:
    learner_id = new_id("pg-course")
    c.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(DEMO_LEARNER["profile"]))
    return learner_id


async def test_a_course_goal_and_curriculum_are_shared_across_processes(tmp_path) -> None:
    api, worker = container(tmp_path / "data"), container(tmp_path / "data")
    try:
        learner_id = learner(api)
        goal, _ = await api.curriculum_service.create_goal(
            learner_id, GoalInput(title="Reach B1", domain="spanish", target_level="B1"))
        assert set(goal.target_concepts) == ALL
        build = await api.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
        assert build.status == TaskStatus.COMPLETED, build.errors

        mine, theirs = api.curriculum_service.curriculum(goal.goal_id), worker.curriculum_service.curriculum(goal.goal_id)
        assert theirs.version_id == mine.version_id and theirs.version == 1
        assert [o.concept_id for o in theirs.objectives] == [o.concept_id for o in mine.objectives]
        assert {o.concept_id for o in theirs.objectives} == ALL

        action = await worker.curriculum_service.next_action(learner_id)
        assert action.action == LearningActionType.LEARN
        again = await api.curriculum_service.next_action(learner_id)
        assert (again.objective_id, again.concept_id, again.curriculum_version) == \
            (action.objective_id, action.concept_id, action.curriculum_version)
        lesson = await answer_all(worker, await worker.curriculum_service.start_lesson(action, user_id="u1"))
        assert lesson.status == TaskStatus.COMPLETED, lesson.errors
        assert api.task_service.get(lesson.task_id).status == TaskStatus.COMPLETED
    finally:
        api.close()
        worker.close()


def test_concurrent_builds_of_a_course_curriculum_store_one_version(tmp_path) -> None:
    containers = [container(tmp_path / "data") for _ in range(PROCESSES)]
    try:
        learner_id = learner(containers[0])
        goal, _ = asyncio.run(containers[0].curriculum_service.create_goal(
            learner_id, GoalInput(title="Reach A2", domain="spanish", target_level="A2")))
        start = threading.Barrier(PROCESSES)
        results, errors = [], []

        def build(c) -> None:
            try:
                start.wait()
                results.append(asyncio.run(c.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")))
            except Exception as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=build, args=(c,)) for c in containers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({t.task_id for t in results}) <= 1, "one planning task for one goal"
        versions = containers[0].curriculum_service.engine.versions(curriculum_id_for(goal.goal_id))
        assert len(versions) == 1 and len(versions[0].objectives) == len(goal.target_concepts)
        completed = [t for t in results if t.status == TaskStatus.COMPLETED]
        # the others found the one planning task leased by another process: busy, never a second task or version
        assert completed and all(isinstance(e, TaskBusy) for e in errors), (results, errors)
        assert len(results) + len(errors) == PROCESSES
    finally:
        for c in containers:
            c.close()
