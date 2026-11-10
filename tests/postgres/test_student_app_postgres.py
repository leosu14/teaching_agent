"""The student web app's API path on PostgreSQL (MVP item 6). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to.

The app talks to whichever API replica the load balancer picks, so every request of a journey may land on another
process. Two API containers share one database: the learner's profile, goal and plan are made on one; the cycle is
started on the other; its prompts are answered alternately on both; the lesson reading is built on the replica that
did not start the session, and still holds back the session's open answer, because the open question is read from the
database, not from process memory. Every run uses fresh ids, so the database can be reused."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.common import new_id
from app.schemas.learning_cycle import LearnerPrompt
from app.schemas.learner import LearnerProfileInput
from app.services.container import build_container
from app.utils.text import reveals
from scripts.run_learning_cycle_demo import ScriptedLearner, scenario, settings
from tests.auth import TestAuth, client_for

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_curriculum_demo as curriculum_demo  # noqa: E402

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL student app tests")


def replica(data_dir: Path):
    return build_container(settings(data_dir, database_url=URL, log_json=False),
                           llm_providers={"mock": MockLLMProvider(default_responders())})


def placed_learner(c) -> str:
    """The learning-cycle demo's learner (placement evidence, secured concepts) under a fresh id."""
    fx = curriculum_demo._fixture("learner.json")
    learner_id = new_id("pg-web")
    c.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(fx["profile"]))

    async def place() -> None:
        for domain, evidence in curriculum_demo.placement_evidence(learner_id, fx["placement"]).items():
            await c.learner_service.record_evidence(learner_id, domain, evidence)
        for concept in scenario()["secured_concepts"]:
            await c.learner_service.record_evidence(
                learner_id, "spanish", curriculum_demo.exercise_evidence(learner_id, concept, 4, "secured"))

    asyncio.run(place())
    return learner_id


def test_a_journey_across_two_api_replicas(tmp_path) -> None:
    first, second = replica(tmp_path / "data"), replica(tmp_path / "data")
    try:
        learner_id = placed_learner(first)
        auth = TestAuth()
        with client_for(first, learner_id, auth=auth) as a, client_for(second, learner_id, auth=auth) as b:
            assert a.get("/me").json()["learner_ids"] == [learner_id]
            goal = a.post(f"/learners/{learner_id}/goals", json={
                "title": "Reach B1 Spanish", "domain": "spanish", "target_level": "B1",
                "idempotency_key": new_id("goal"), "metadata": {"framework_id": "cefr"}})
            assert goal.status_code == 201, goal.text
            assert b.get(f"/goals/{goal.json()['goal_id']}/curriculum").status_code == 404
            assert a.post(f"/goals/{goal.json()['goal_id']}/curriculum").status_code == 201
            assert b.get(f"/learners/{learner_id}/next-action").json()["action"] == "LEARN"

            key = new_id("web")
            view = b.post(f"/learners/{learner_id}/learning-cycles", json={"idempotency_key": key}).json()
            assert view["prompt"]["kind"] == "DIAGNOSTIC_QUESTIONS"
            again = a.post(f"/learners/{learner_id}/learning-cycles", json={"idempotency_key": key})
            assert again.status_code == 200 and again.json()["cycle_id"] == view["cycle_id"]  # a retried Start

            learner, clients, turn = ScriptedLearner(prefix="pg"), [a, b], 0
            lesson_checked = False
            for _ in range(40):
                if view["status"] != "WAITING":
                    break
                if view["prompt"]["kind"] == "SESSION_QUESTION" and not lesson_checked:
                    lesson_checked = True
                    session_id = next(s["child_id"] for s in view["steps"] if s["kind"] == "TEACHING_SESSION")
                    question = first.teaching_service.repository.get(session_id).state.pending_question
                    reader = clients[(turn + 1) % 2]  # not the replica that just answered
                    lesson = next(s["child_id"] for s in view["steps"] if s["kind"] == "LESSON")
                    reading = reader.get(f"/lessons/{lesson}/reading")
                    assert reading.status_code == 200, reading.text
                    assert reading.json()["sections"]
                    open_answers = [question.expected_answer, *question.accepted_answers]
                    texts = [p for s in reading.json()["sections"] for p in [*s["paragraphs"], *s["examples"]]]
                    assert not [t for t in texts if reveals(t, open_answers)]
                response = learner.respond(LearnerPrompt.model_validate(view["prompt"]))
                r = clients[turn % 2].post(f"/learning-cycles/{view['cycle_id']}/responses",
                                           json=response.model_dump(mode="json", exclude_none=True))
                assert r.status_code == 200, r.text
                view, turn = r.json(), turn + 1
            assert lesson_checked and view["status"] == "COMPLETED", view["status"]

            [listed] = a.get(f"/learners/{learner_id}/learning-cycles").json()
            assert listed["cycle_id"] == view["cycle_id"] and listed["status"] == "COMPLETED"
            progress = b.get(f"/learners/{learner_id}/progress").json()
            assert view["action"]["concept_id"] in progress["mastered"]
            assert a.get(f"/learners/{learner_id}/next-action").json()["concept_id"] != view["action"]["concept_id"]
    finally:
        first.close()
        second.close()
