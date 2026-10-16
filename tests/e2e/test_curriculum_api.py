"""Goals, curricula and the next learning action over HTTP."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.services.container import build_container
from scripts.run_curriculum_demo import FIXTURES, placement_evidence
from tests.auth import client_for

FIXTURE = json.loads((FIXTURES / "learner.json").read_text(encoding="utf-8"))
LEARNER = FIXTURE["learner_id"]


@pytest.fixture
def client(tmp_path):
    container = build_container(Settings(data_dir=tmp_path / "data", corpus_dir=FIXTURES, log_json=False),
                                llm_providers={"mock": MockLLMProvider(default_responders())})
    with client_for(container, LEARNER) as c:
        c.container = container
        yield c
    container.close()


async def _place(container) -> None:
    for domain, evidence in placement_evidence(LEARNER, FIXTURE["placement"]).items():
        await container.learner_service.record_evidence(LEARNER, domain, evidence)


def test_goals_curriculum_and_next_action_over_http(client: TestClient) -> None:
    assert client.put(f"/learners/{LEARNER}", json=FIXTURE["profile"]).status_code == 200
    client.portal.call(_place, client.container)
    body = {k: v for k, v in FIXTURE["goal"].items()}

    created = client.post(f"/learners/{LEARNER}/goals", json=body)
    assert created.status_code == 201, created.text
    goal = created.json()
    assert goal["status"] == "ACTIVE" and goal["target_source"] == "level"
    again = client.post(f"/learners/{LEARNER}/goals", json=body)
    assert again.status_code == 200 and again.json()["goal_id"] == goal["goal_id"]
    assert [g["goal_id"] for g in client.get(f"/learners/{LEARNER}/goals").json()] == [goal["goal_id"]]
    assert client.get(f"/goals/{goal['goal_id']}").json()["title"] == "Reach B1 Spanish"

    assert client.get("/goals/goal_unknown").status_code == 404
    assert client.get(f"/goals/{goal['goal_id']}/curriculum").status_code == 404  # not built yet
    bad = client.post(f"/learners/{LEARNER}/goals",
                      json={"title": "x", "domain": "spanish", "target_concepts": ["es.invented"]})
    assert bad.status_code == 422 and "es.invented" in bad.json()["detail"]
    assert client.post(f"/learners/{LEARNER}/goals", json={"title": "x", "domain": "spanish"}).status_code == 422

    build = client.post(f"/goals/{goal['goal_id']}/curriculum")
    assert build.status_code == 201, build.text
    built = build.json()
    assert built["status"] == "COMPLETED" and built["new_version"] and built["curriculum"]["version"] == 1
    rebuild = client.post(f"/goals/{goal['goal_id']}/curriculum")
    assert rebuild.status_code == 200 and not rebuild.json()["new_version"]

    cur = client.get(f"/goals/{goal['goal_id']}/curriculum").json()
    assert [o["concept_id"] for o in cur["objectives"]] == ["es.present", "es.preterite", "es.past_contrast",
                                                             "es.subjunctive"]
    assert cur["progress"]["required"] == 4
    versions = client.get(f"/goals/{goal['goal_id']}/curriculum/versions").json()
    assert [v["version"] for v in versions] == [1]

    action = client.get(f"/learners/{LEARNER}/next-action")
    assert action.status_code == 200
    assert (action.json()["action"], action.json()["concept_id"]) == ("LEARN", "es.preterite")
    assert client.get("/learners/nobody/next-action").status_code == 404

    changed = client.patch(f"/goals/{goal['goal_id']}", json={"target_level": "B2"})
    assert changed.status_code == 200, changed.text
    assert changed.json()["curriculum"]["version"] == 2 and "es.argument" in changed.json()["goal"]["target_concepts"]
    versions = client.get(f"/goals/{goal['goal_id']}/curriculum/versions").json()
    assert [v["version"] for v in versions] == [1, 2] and versions[0]["version_id"] != versions[1]["version_id"]
    assert client.patch(f"/goals/{goal['goal_id']}", json={"status": "COMPLETED"}).status_code == 422

    # No provider details or secrets in any response.
    for path in (f"/goals/{goal['goal_id']}/curriculum", f"/learners/{LEARNER}/next-action",
                 f"/goals/{goal['goal_id']}/curriculum/versions"):
        text = client.get(path).text
        assert "mock" not in text and "api_key" not in text and "provider" not in text, path
