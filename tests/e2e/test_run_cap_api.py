"""The run-wide cap end to end over HTTP (MVP item 10): the live smoke run's flow and limits on the mock providers,
with every paid capability priced as in the rehearsal (deploy/live_smoke.py --mock, compose.live.yaml).

One lesson (run by the worker) and one graded teaching-session answer are charged, every paid attempt of both, to
the one `total:all` scope; its total stays under the cap, nothing is left reserved, TTS, images and search are priced.
With a cap too small for the lesson, the lesson stops on the run-wide budget before the request that would pass it,
and nothing more is charged."""

from __future__ import annotations

import pytest

from app.config.production import ProductionSettings
from app.config.providers import ProviderSettings
from app.config.settings import Settings
from app.observability.spend import TOTAL_SCOPE
from app.schemas.spend import SpendState
from app.services.container import build_container
from tests.auth import client_for, settle
from tests.conftest import LEARNER
from tests.e2e.test_api import answer

# compose.live.yaml's unit limits and the rehearsal's prices (deploy/live_smoke.py, MOCK_PRICES)
LIMITS = ProductionSettings(max_llm_requests=40, max_llm_tokens=300_000, max_search_requests=8,
                            max_generated_images=1, max_searched_images=0, max_tts_characters=8_000,
                            max_tts_seconds=900, max_generated_video_segments=0)
PRICES = {"tts_price_per_mchar": 15.0, "image_price_per_image": 0.04, "search_price_per_request": 0.008}


def capped(tmp_path, mock_llm, cap: float):
    settings = Settings(data_dir=tmp_path / "data", log_json=False, production=LIMITS,
                        providers=ProviderSettings(**PRICES), budget_total_max_cost_usd=cap,
                        budget_session_max_cost_usd=cap, budget_attempt_max_cost_usd=cap,
                        generated_video_enabled=False)
    return build_container(settings, llm_providers={"mock": mock_llm})


def total(container) -> tuple[int, int]:
    t = container.spend.totals([TOTAL_SCOPE])[TOTAL_SCOPE]["cost_microusd"]
    return t.reserved, t.settled


def entries(container) -> list:
    return [e for e in container.spend.entries() if e.state in (SpendState.SETTLED, SpendState.UNKNOWN)]


def test_a_lesson_and_a_graded_answer_are_charged_to_the_cap_and_stay_under_it(tmp_path, mock_llm) -> None:
    container = capped(tmp_path, mock_llm, 2.0)
    lid = LEARNER["learner_id"]
    try:
        with client_for(container, lid) as client:
            assert client.put(f"/learners/{lid}", json=LEARNER["profile"]).status_code == 200
            task = settle(client, client.post("/tasks", json={"request": LEARNER["request"], "learner_id": lid}))
            while task["status"] == "WAITING":
                task = settle(client, client.post(f"/tasks/{task['task_id']}/answers",
                                                  json={"answers": answer(task)}))
            assert task["status"] == "COMPLETED", task.get("error")
            started = client.post(f"/lessons/{task['task_id']}/teaching-session", json={})
            assert started.status_code == 201, started.text
            question = started.json()["waiting_question"]
            reply = question["choices"][0] if question["choices"] else "Hola, me llamo Ana."
            graded = client.post(f"/teaching-sessions/{started.json()['session_id']}/answers",
                                 json={"answer": reply, "client_turn_id": "c1"})
            assert graded.status_code == 200, graded.text
            assert graded.json()["answer_outcome"] or graded.json()["correct"] is not None
        paid = entries(container)
        assert paid and all(TOTAL_SCOPE in e.scope_keys for e in paid)  # nothing paid outside the cap
        assert {e.session_id for e in paid} - {None}  # the session's teacher turns and grading are in it too
        reserved, settled = total(container)
        assert reserved == 0 and 0 < settled <= 2_000_000
        assert settled == sum(e.final_cost_microusd or 0 for e in paid)
        by_capability = {e.capability for e in paid}
        assert {"llm", "tts", "image", "search"} <= by_capability
        assert all(e.cost_source == "priced" for e in paid if e.capability == "llm")  # reconciled from usage
        for e in paid:
            assert e.cost_source != "unpriced", e
            if e.capability in ("tts", "image", "search"):
                assert e.cost_source == "priced" and e.final_cost_microusd > 0
    finally:
        container.close()


@pytest.mark.parametrize("cap", [0.01])
def test_a_cap_too_small_for_the_lesson_stops_it_before_the_request_that_would_pass_it(tmp_path, mock_llm,
                                                                                     cap: float) -> None:
    container = capped(tmp_path, mock_llm, cap)
    lid = LEARNER["learner_id"]
    try:
        with client_for(container, lid) as client:
            assert client.put(f"/learners/{lid}", json=LEARNER["profile"]).status_code == 200
            task = settle(client, client.post("/tasks", json={"request": LEARNER["request"], "learner_id": lid}))
            while task["status"] == "WAITING":
                task = settle(client, client.post(f"/tasks/{task['task_id']}/answers",
                                                  json={"answers": answer(task)}))
            assert task["status"] == "FAILED"
            assert {e["category"] for e in task["errors"]} == {"BudgetExceededError"}, task["errors"]
        raw = container.task_service.get(task["task_id"]).errors
        assert any("run-wide" in e.message and "never resets" in e.message for e in raw), raw
        reserved, settled = total(container)
        assert reserved == 0 and settled <= cap * 1_000_000  # refused before dispatch: never past the cap
    finally:
        container.close()
