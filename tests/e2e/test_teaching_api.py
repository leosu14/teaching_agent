"""Interactive teaching sessions over HTTP."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from tests.auth import client_for
from tests.teaching_fixtures import KEY, Env


@pytest.fixture
def client(teaching_env: Env):
    learner = teaching_env.container.task_service.get(teaching_env.lesson_task).learner_id
    with client_for(teaching_env.container, learner) as c:
        c.env = teaching_env
        yield c


def _answer(client: TestClient, sid: str, say: str, cid: str):
    question = client.get(f"/teaching-sessions/{sid}").json()["state"]["waiting_question"]
    text = next(a[say] for cue, a in KEY.items() if cue in question["prompt"])
    return client.post(f"/teaching-sessions/{sid}/answers", json={"answer": text, "client_turn_id": cid})


def test_a_session_over_http(client: TestClient) -> None:
    task_id = client.env.lesson_task
    lesson_artifact = client.env.container.artifacts.find(task_id, "lesson").artifact_id

    created = client.post(f"/lessons/{lesson_artifact}/teaching-session", json={"idempotency_key": "web"})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["status"] == "WAITING_FOR_LEARNER" and body["difficulty"] == 2 and body["created"]
    assert body["first_turn"]["turn_type"] == "EXPLANATION" and body["objective"]["concept_id"] == "es.past_contrast"
    assert body["objective_id"] and body["waiting_question"]["prompt"]
    assert "expected_answer" not in created.text  # the answer key never leaves the service
    sid = body["session_id"]
    again = client.post(f"/lessons/{task_id}/teaching-session", json={"idempotency_key": "web"})
    assert again.status_code == 200 and again.json()["session_id"] == sid  # by lesson task id, same session

    view = client.get(f"/teaching-sessions/{sid}")
    assert view.status_code == 200
    v = view.json()
    assert v["status"] == "WAITING_FOR_LEARNER" and v["next_action"]["kind"] == "answer_question"
    assert [t["sequence"] for t in v["turns"]] == [1, 2] and v["objective_progress"]["concept_id"] == "es.past_contrast"
    for internal in ("expected_answer", "accepted_answers", "provider", "config_fingerprint", "input_payload"):
        assert internal not in view.text, internal

    wrong = _answer(client, sid, "incorrect", "w1")
    assert wrong.status_code == 200, wrong.text
    w = wrong.json()
    assert w["correct"] is False and w["difficulty_change"]["after"] == 1
    assert [t["turn_type"] for t in w["teacher_turns"]] == ["HINT"]
    duplicate = client.post(f"/teaching-sessions/{sid}/answers",
                            json={"answer": w["learner_turn"]["content"], "client_turn_id": "w1"})
    assert duplicate.status_code == 200 and duplicate.json()["replayed"]
    assert duplicate.json()["learner_turn"]["turn_id"] == w["learner_turn"]["turn_id"]
    reused = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": "otra", "client_turn_id": "w1"})
    assert reused.status_code == 409

    asked = client.post(f"/teaching-sessions/{sid}/answers",
                        json={"answer": "What is the capital of Australia?", "kind": "question",
                              "client_turn_id": "q1"})
    assert asked.status_code == 200 and asked.json()["teacher_turns"][0]["metadata"]["grounded"] is False

    paused = client.post(f"/teaching-sessions/{sid}/pause")
    assert paused.status_code == 200 and paused.json()["status"] == "PAUSED"
    assert client.post(f"/teaching-sessions/{sid}/pause").status_code == 200
    refused = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": "sonó", "client_turn_id": "p"})
    assert refused.status_code == 409 and "paused" in refused.json()["detail"]
    resumed = client.post(f"/teaching-sessions/{sid}/resume")
    assert resumed.status_code == 200 and resumed.json()["status"] == "WAITING_FOR_LEARNER"

    for i, say in enumerate(["correct", "correct", "correct"]):
        r = _answer(client, sid, say, f"c{i}")
        assert r.status_code == 200, r.text
    done = r.json()
    assert done["status"] == "COMPLETED" and done["completion_reason"] == "OBJECTIVE_DEMONSTRATED"
    assert done["summary"]["difficulty_trajectory"] == [2, 1, 2] and done["outcome"]["mastery_changes"]
    final = client.get(f"/teaching-sessions/{sid}").json()
    assert final["next_action"]["kind"] == "learning_action" and final["next_action"]["learning_action"]["action"]
    assert final["summary"]["completion_reason"] == "OBJECTIVE_DEMONSTRATED"

    assert client.post(f"/teaching-sessions/{sid}/answers", json={"answer": "x"}).status_code == 409
    assert client.post(f"/teaching-sessions/{sid}/pause").status_code == 409
    assert client.post(f"/teaching-sessions/{sid}/cancel").status_code == 409


def test_cancel_over_http(client: TestClient) -> None:
    sid = client.post(f"/lessons/{client.env.lesson_task}/teaching-session", json={}).json()["session_id"]
    first = client.post(f"/teaching-sessions/{sid}/cancel")
    assert first.status_code == 200 and first.json()["status"] == "CANCELLED"
    second = client.post(f"/teaching-sessions/{sid}/cancel")
    assert second.status_code == 200 and second.json()["turns"] == first.json()["turns"]
    assert client.post(f"/teaching-sessions/{sid}/resume").status_code == 409


def test_errors_over_http(client: TestClient) -> None:
    assert client.get("/teaching-sessions/tsess_missing").status_code == 404
    assert client.post("/teaching-sessions/tsess_missing/answers", json={"answer": "x"}).status_code == 404
    assert client.post("/teaching-sessions/tsess_missing/pause").status_code == 404
    assert client.post("/lessons/art_missing/teaching-session", json={}).status_code == 404
    bad = client.post(f"/lessons/{client.env.lesson_task}/teaching-session",
                      json={"objective_id": "obj_elsewhere"})
    assert bad.status_code == 422
    assert client.post(f"/lessons/{client.env.lesson_task}/teaching-session",
                       json={"mastery": 1.0}).status_code == 422  # unknown fields are rejected
    sid = client.post(f"/lessons/{client.env.lesson_task}/teaching-session", json={}).json()["session_id"]
    assert client.post(f"/teaching-sessions/{sid}/answers", json={"answer": ""}).status_code == 422
    assert client.post(f"/teaching-sessions/{sid}/answers",
                       json={"answer": "x", "kind": "complete_session"}).status_code == 422


def test_concurrent_answers_over_http_are_applied_once_or_refused(client: TestClient) -> None:
    sid = client.post(f"/lessons/{client.env.lesson_task}/teaching-session", json={}).json()["session_id"]
    question = client.get(f"/teaching-sessions/{sid}").json()["state"]["waiting_question"]
    text = next(a["correct"] for cue, a in KEY.items() if cue in question["prompt"])

    def send(cid: str):
        return client.post(f"/teaching-sessions/{sid}/answers", json={"answer": text, "client_turn_id": cid})

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(send, ["a", "b", "c", "d"]))
    assert {r.status_code for r in responses} <= {200, 409}
    applied = [r.json() for r in responses if r.status_code == 200]
    turns = client.get(f"/teaching-sessions/{sid}").json()["turns"]
    answers = [t for t in turns if t["turn_type"] == "LEARNER_ANSWER"]
    # every accepted answer is stored exactly once; every refused one was told so
    assert sorted(t["turn_id"] for t in answers) == sorted(r["learner_turn"]["turn_id"] for r in applied)
    assert [t["sequence"] for t in turns] == list(range(1, len(turns) + 1))
