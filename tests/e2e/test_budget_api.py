"""Budget exhaustion over HTTP (PR19): a teacher turn or the grading of an attempt that its budget has no room for
answers 429 with fixed text, sends nothing to the provider, and stays distinct from a 503 (teacher unavailable)."""

from __future__ import annotations

from app.api.auth import AUTHOR
from app.services.budgets import BudgetExhausted
from tests.auth import client_for
from tests.e2e.test_assessment_api import item_body
from tests.teaching_fixtures import KEY, open_env

AGENT = "teaching_session"
EXHAUSTED = {"error": "BudgetExhausted", "detail": BudgetExhausted.MESSAGE}


def test_a_teaching_session_without_budget_answers_429(teaching_env, tmp_path) -> None:
    template = teaching_env.data_dir
    learner = teaching_env.container.task_service.get(teaching_env.lesson_task).learner_id
    with client_for(teaching_env.container, learner) as client:
        sid = client.post(f"/lessons/{teaching_env.lesson_task}/teaching-session",
                          json={"idempotency_key": "b1"}).json()["session_id"]
        question = client.get(f"/teaching-sessions/{sid}").json()["state"]["waiting_question"]
    wrong = next(a["incorrect"] for cue, a in KEY.items() if cue in question["prompt"])
    teaching_env.close()

    # The same data, reopened with a session budget that is already used up.
    env = open_env(tmp_path / "tight", teaching_env.lesson_task, copy_from=template,
                   budget_session_max_llm_requests=0)
    try:
        with client_for(env.container, learner) as client:
            calls = env.llm.calls[AGENT]
            r = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": wrong, "client_turn_id": "w1"})
            assert r.status_code == 429, r.text
            assert r.json() == EXHAUSTED
            assert env.llm.calls[AGENT] == calls  # nothing was sent
            assert [e for e in env.events if e.type == "budget.exhausted" and e.data["budget"] == "session"]
            view = client.get(f"/teaching-sessions/{sid}").json()
            assert view["status"] == "ACTIVE" and view["next_action"]["kind"] == "teacher_reply_pending"
    finally:
        env.close()


def test_grading_an_attempt_without_budget_answers_429(tmp_path, teaching_lesson) -> None:
    template, task_id = teaching_lesson
    env = open_env(tmp_path / "data", task_id, copy_from=template, budget_attempt_max_llm_requests=0)
    try:
        learner = env.container.task_service.get(task_id).learner_id
        with client_for(env.container, learner, roles=(AUTHOR,)) as client:
            client.env = env
            body, answers = item_body(client)
            item_id = client.post("/assessment-items", json=body).json()["assessment_item_id"]
            r = client.post(f"/assessment-items/{item_id}/attempts",
                            json={"learner_id": learner, "answer": answers["semantic"], "attempt_id": "b-1"})
            assert r.status_code == 429, r.text
            assert r.json() == EXHAUSTED
            # Nothing was graded or stored: the same attempt can be submitted again once the budget allows.
            assert client.get("/assessment-attempts/b-1").status_code == 404
    finally:
        env.close()
