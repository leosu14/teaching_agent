"""Learning cycles over HTTP."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.schemas.learning_cycle import LearnerPrompt
from tests.learning_cycle_fixtures import LEARNER, ScriptedLearner


def client_at(cycle_env_at, stage: str) -> TestClient:
    return TestClient(create_app(cycle_env_at(stage).container))


def answer_over_http(client: TestClient, body: dict, learner: ScriptedLearner) -> dict:
    for _ in range(40):
        if body["status"] != "WAITING":
            return body
        response = learner.respond(LearnerPrompt.model_validate(body["prompt"]))
        reply = client.post(f"/learning-cycles/{body['cycle_id']}/responses",
                            json=response.model_dump(mode="json", exclude_none=True))
        assert reply.status_code == 200, reply.text
        body = reply.json()
    raise AssertionError("the cycle did not finish")


def test_a_cycle_over_http(cycle_env_at) -> None:
    with client_at(cycle_env_at, "base") as client:
        created = client.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "web"})
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["created"] and body["status"] == "WAITING" and body["action"]["action"] == "LEARN"
        assert body["prompt"]["kind"] == "DIAGNOSTIC_QUESTIONS"
        cid = body["cycle_id"]

        again = client.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "web"})
        assert again.status_code == 200 and again.json()["cycle_id"] == cid and not again.json()["created"]
        other = client.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "other"})
        assert other.status_code == 409
        changed = client.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "web",
                                                                            "user_id": "someone-else"})
        assert changed.status_code == 409  # the same key with a different request

        wrong = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "w", "answer": "x"})
        assert wrong.status_code == 422  # a sheet is asked for, not a session answer
        assert client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "w"}).status_code == 422

        done = answer_over_http(client, body, ScriptedLearner(prefix="h"))
        assert done["status"] == "COMPLETED" and done["outcome"]["result"] == "TAUGHT"
        assert done["outcome"]["learning_evidence_ids"] and done["outcome"]["next_action"]
        assert "expected_answer" not in str(done)

        assert client.get(f"/learning-cycles/{cid}").json()["status"] == "COMPLETED"
        listed = client.get(f"/learners/{LEARNER}/learning-cycles").json()
        assert [c["cycle_id"] for c in listed] == [cid]
        events = client.get(f"/learning-cycles/{cid}/events").json()
        assert {"learning_cycle.started", "learning_cycle.completed"} <= {e["type"] for e in events}
        assert all(LEARNER not in str(e) for e in events)
        artifacts = client.get(f"/learning-cycles/{cid}/artifacts").json()
        assert {a["name"] for a in artifacts} == {"learning_cycle", "learning_cycle_outcome"}

        late = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "late", "answer": "x"})
        assert late.status_code == 409
        assert client.post(f"/learning-cycles/{cid}/cancel").status_code == 409
        assert client.post(f"/learning-cycles/{cid}/resume").json()["status"] == "COMPLETED"


def test_unknown_cycles_are_404(cycle_env_at) -> None:
    with client_at(cycle_env_at, "base") as client:
        assert client.get("/learning-cycles/lcyc_missing").status_code == 404
        assert client.post("/learning-cycles/lcyc_missing/resume").status_code == 404
        assert client.post("/learning-cycles/lcyc_missing/responses",
                           json={"client_response_id": "r", "answer": "x"}).status_code == 404


def test_cancel_over_http_frees_the_learner(cycle_env_at) -> None:
    with client_at(cycle_env_at, "session") as client:
        [listed] = client.get(f"/learners/{LEARNER}/learning-cycles").json()
        assert listed["status"] == "WAITING" and listed["prompt"] is None  # prompts only on the cycle itself
        cycle = client.get(f"/learning-cycles/{listed['cycle_id']}").json()
        assert cycle["prompt"]["kind"] == "SESSION_QUESTION"
        cancelled = client.post(f"/learning-cycles/{cycle['cycle_id']}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "CANCELLED"
        assert client.post(f"/learning-cycles/{cycle['cycle_id']}/cancel").status_code == 200
        assert client.post(f"/learning-cycles/{cycle['cycle_id']}/resume").status_code == 409
