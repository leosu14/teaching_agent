"""Assessment items and graded attempts over HTTP."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from scripts.run_assessment_demo import fixture
from app.api.auth import AUTHOR
from tests.auth import client_for
from tests.teaching_fixtures import Env


@pytest.fixture
def client(teaching_env: Env):
    learner = teaching_env.container.task_service.get(teaching_env.lesson_task).learner_id
    with client_for(teaching_env.container, learner, roles=(AUTHOR,)) as c:
        c.env = teaching_env
        yield c


def item_body(client: TestClient, key: str = "why_preterite") -> tuple[dict, dict]:
    entry = next(e for e in fixture()["items"] if e["item"]["assessment_item_id"] == key)
    body = {"item": {**entry["item"], "lesson_id": client.env.lesson_task}}
    if "rubric" in entry:
        body["rubric"] = entry["rubric"]
    return body, entry["answers"]


def test_items_and_attempts_over_http(client: TestClient) -> None:
    body, answers = item_body(client)
    learner = client.env.container.task_service.get(client.env.lesson_task).learner_id
    created = client.post("/assessment-items", json=body)
    assert created.status_code == 201, created.text
    item_id = created.json()["assessment_item_id"]
    assert client.post("/assessment-items", json=body).status_code == 200  # the same item again
    changed = json.loads(json.dumps(body))
    changed["item"]["expected_answer"] = "Something else."
    assert client.post("/assessment-items", json=changed).status_code == 409

    first = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": answers["semantic"], "attempt_id": "web-1"})
    assert first.status_code == 201, first.text
    result = first.json()
    assert result["grade"]["outcome"] == "CORRECT" and result["grade"]["grader_type"] == "SEMANTIC"
    assert result["attempt"]["attempt_number"] == 1 and result["attempt"]["mastery_updated"]
    assert result["mastery_changes"] and result["learning_action"]["action"]
    assert "expected_answer" not in first.text and "The action is completed" not in first.text
    assert "llm_calls" not in first.text and "input_tokens" not in first.text  # no provider internals

    replay = client.post(f"/assessment-items/{item_id}/attempts",
                         json={"learner_id": learner, "answer": answers["semantic"], "attempt_id": "web-1"})
    assert replay.status_code == 200 and replay.json()["replayed"]
    assert replay.json()["grade"] == result["grade"]
    conflict = client.post(f"/assessment-items/{item_id}/attempts",
                           json={"learner_id": learner, "answer": answers["partial"], "attempt_id": "web-1"})
    assert conflict.status_code == 409 and conflict.json()["error"] == "AttemptConflict"

    unsure = client.post(f"/assessment-items/{item_id}/attempts",
                         json={"learner_id": learner, "answer": answers["uncertain"]})
    assert unsure.status_code == 201
    u = unsure.json()
    assert u["grade"]["outcome"] == "UNCERTAIN" and u["attempt"]["retry_recommended"]
    assert u["attempt"]["attempt_number"] == 2 and not u["attempt"]["mastery_updated"] and not u["mastery_changes"]

    attempt = client.get("/assessment-attempts/web-1")
    assert attempt.status_code == 200 and attempt.json()["outcome"] == "CORRECT" and attempt.json()["completed"]
    grade = client.get("/assessment-attempts/web-1/grade")
    assert grade.status_code == 200 and grade.json() == result["grade"]
    listed = client.get(f"/assessment-items/{item_id}/attempts", params={"learner_id": learner})
    assert [a["outcome"] for a in listed.json()] == ["CORRECT", "UNCERTAIN"]


def test_errors_over_http(client: TestClient) -> None:
    body, answers = item_body(client, "conjugate")
    learner = client.env.container.task_service.get(client.env.lesson_task).learner_id
    item_id = client.post("/assessment-items", json=body).json()["assessment_item_id"]
    assert client.get("/assessment-attempts/nope").status_code == 404
    assert client.get("/assessment-attempts/nope/grade").status_code == 404
    assert client.post("/assessment-items/nope/attempts", json={"learner_id": learner, "answer": "x"}).status_code \
        == 404
    assert client.post(f"/assessment-items/{item_id}/attempts",
                       json={"learner_id": "nobody", "answer": "x"}).status_code == 404
    assert client.post(f"/assessment-items/{item_id}/attempts", json={"learner_id": learner}).status_code == 422
    other = json.loads(json.dumps(body))
    other["item"].update(assessment_item_id="other", concept_id="es.subjunctive")
    assert client.post("/assessment-items", json=other).status_code == 422
    bad_rubric = json.loads(json.dumps(item_body(client)[0]))
    bad_rubric["rubric"]["criteria"][0]["weight"] = 0.9  # weights no longer sum to 1
    assert client.post("/assessment-items", json=bad_rubric).status_code == 422
    exact = client.post(f"/assessment-items/{item_id}/attempts", json={"learner_id": learner,
                                                                        "answer": answers["exact"]})
    assert exact.json()["grade"]["grader_type"] == "EXACT" and exact.json()["grade"]["outcome"] == "CORRECT"
