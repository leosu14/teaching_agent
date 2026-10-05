"""API end-to-end: the full lesson slice over HTTP, using the same services as the CLI demo."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from tests.conftest import ANSWERS, EVALUATION_ANSWERS, LEARNER


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c:
        yield c


def answer(task: dict) -> list[dict]:
    sheet = task["waiting"]["prompt"]
    key = ANSWERS["rounds"][sheet["round_number"] - 1]
    return [{"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]


def test_lesson_task_over_http(client: TestClient) -> None:
    learner_id = LEARNER["learner_id"]
    assert client.get("/health").json() == {"status": "ok"}
    assert client.put(f"/learners/{learner_id}", json=LEARNER["profile"]).status_code == 200

    created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": learner_id})
    assert created.status_code == 201
    task = created.json()
    assert task["status"] == "WAITING"
    assert task["waiting"]["kind"] == "diagnostic_answers"
    assert all("expected_answer" not in q for q in task["waiting"]["prompt"]["questions"])  # no answer keys leak

    rounds = 0
    while task["status"] == "WAITING":
        rounds += 1
        resp = client.post(f"/learners/{learner_id}/assessment",
                           json={"task_id": task["task_id"], "answers": answer(task)})
        assert resp.status_code == 200, resp.text
        task = resp.json()
    assert rounds == 2
    assert task["status"] == "COMPLETED"

    fetched = client.get(f"/tasks/{task['task_id']}").json()
    assert fetched["result"]["review_verdict"] == "APPROVED"
    assert fetched["cost"]["actual_cost_usd"] > 0 and fetched["cost"]["estimated_cost_usd"] > 0

    artifacts = client.get(f"/tasks/{task['task_id']}/artifacts").json()
    assert {a["type"] for a in artifacts} == {"LEARNING_EVIDENCE", "LEARNER_MODEL", "KNOWLEDGE_GAPS",
                                             "PEDAGOGICAL_PLAN", "RESEARCH_BUNDLE", "VISUAL_PLAN", "IMAGE_ASSET",
                                                "LESSON_PLAN", "LESSON", "SCRIPT", "SLIDE_PLAN", "REPORT", "PRESENTATION",
                                                "AUDIO_PLAN", "AUDIO_ASSET", "PRESENTATION_TIMELINE",
                                                "VIDEO_PLAN", "SUBTITLE", "VIDEO"}

    events = client.get(f"/tasks/{task['task_id']}/events").json()
    types = [e["type"] for e in events]
    assert types[0] == "task.created" and types[-1] == "task.completed"

    progress = client.get(f"/learners/{learner_id}/progress").json()
    assert progress["lessons_completed"] == 1 and progress["assessments_taken"] == 1

    assert client.post(f"/tasks/{task['task_id']}/cancel").status_code == 409


def test_catalog_endpoints(client: TestClient) -> None:
    agents = {a["id"] for a in client.get("/agents").json()}
    assert {"request_interpreter", "knowledge_diagnostic", "research", "curriculum_planner",
            "teacher", "content_reviewer", "slide_planner"} <= agents
    tools = {t["name"] for t in client.get("/tools").json()}
    assert {"search.web", "rag.retrieve", "learner.snapshot", "artifact.store", "video.compose"} <= tools
    providers = client.get("/providers").json()
    assert providers["llm_providers"] == ["mock"] and "cefr" in providers["level_frameworks"]


def test_errors_map_to_http_status(client: TestClient) -> None:
    assert client.get("/tasks/task_missing").status_code == 404
    assert client.get("/learners/nobody/progress").status_code == 404
    assert client.post("/tasks", json={"request": "", "learner_id": "x"}).status_code == 422
    bad = client.put("/learners/l1", json={"subjects": [{"subject": "chess", "framework_id": "elo"}]})
    assert bad.status_code == 422


def test_lesson_evaluation_over_http(client: TestClient) -> None:
    learner_id = LEARNER["learner_id"]
    client.put(f"/learners/{learner_id}", json=LEARNER["profile"])
    lesson = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": learner_id}).json()
    while lesson["status"] == "WAITING":
        lesson = client.post(f"/tasks/{lesson['task_id']}/answers", json={"answers": answer(lesson)}).json()
    assert lesson["status"] == "COMPLETED"  # POST /tasks/{id}/answers also serves the diagnostic

    created = client.post(f"/tasks/{lesson['task_id']}/evaluation", json={"user_id": "u1"})
    assert created.status_code == 201, created.text
    task = created.json()
    assert task["status"] == "WAITING" and task["waiting"]["kind"] == "assessment_answers"
    questions = task["waiting"]["prompt"]["questions"]
    assert questions and all("expected_answer" not in q for q in questions)

    bad = client.post(f"/tasks/{task['task_id']}/answers", json={"answers": [{"question_id": "x", "answer": "y"}]})
    assert bad.status_code == 422
    assert client.get(f"/tasks/{task['task_id']}").json()["status"] == "WAITING"

    key = EVALUATION_ANSWERS["answers"]
    body = {"answers": [{"question_id": q["question_id"], "answer": key[q["concept_id"]][q["kind"]]}
                        for q in questions]}
    done = client.post(f"/tasks/{task['task_id']}/answers", json=body)
    assert done.status_code == 200, done.text
    result = done.json()["result"]
    assert done.json()["status"] == "COMPLETED"
    assert result["remaining_gaps"] == ["es.football.opinions"]
    assert result["recommendation"]["action"] == "reteach"

    artifacts = client.get(f"/tasks/{task['task_id']}/artifacts").json()
    assert {a["type"] for a in artifacts} == {"LEARNER_EVALUATION", "LEARNING_EVIDENCE", "LEARNER_MODEL",
                                              "LEARNING_RECOMMENDATION", "ASSESSMENT_ITEM", "ASSESSMENT_GRADE",
                                              "ASSESSMENT_FEEDBACK"}
    artifact = next(a for a in artifacts if a["type"] == "LEARNER_EVALUATION")
    assert artifact["type"] == "LEARNER_EVALUATION"
    assert artifact["metadata"]["score"] == 0.5
    assert artifact["metadata"]["remaining_gaps"] == ["es.football.opinions"]

    types = [e["type"] for e in client.get(f"/tasks/{task['task_id']}/events").json()]
    for expected in ("assessment.created", "assessment.waiting", "assessment.submitted", "evaluation.started",
                     "evaluation.completed", "learner.mastery_updated", "recommendation.created"):
        assert expected in types, expected

    assert client.post(f"/tasks/{task['task_id']}/answers", json=body).status_code == 409
    assert client.post(f"/tasks/{task['task_id']}/evaluation").status_code == 409
    assert client.post("/tasks/task_missing/evaluation").status_code == 404
