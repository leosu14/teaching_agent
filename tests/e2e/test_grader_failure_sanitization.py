"""PR17 regression: a semantic-grader failure must not leak an answer key to learners.

The semantic grader is given the item's expected answer. When its structured output fails validation on every
attempt, the validation error quotes the model's output (`input_value='...'`), so a grader that echoes the key in a
mistyped field puts the key into the error text. That text is kept on the internal grade (`uncertainty_reason`) and
the `assessment.uncertain` event; the learner-facing `PublicGrade` has no `uncertainty_reason` and an UNCERTAIN
grade's feedback is fixed text. The failure is injected explicitly: every grader attempt returns the canary, the
item's expected answer, as a string `score`."""

from __future__ import annotations

import asyncio
import json

from app.api.auth import AUTHOR
from app.providers.llm.mock_responders import teach
from app.schemas.assessment import AttemptSubmission
from scripts.run_assessment_demo import fixture
from tests.auth import client_for

CANARY = "SECRET_ANSWER_CANARY_7F31"
GRADER = "semantic_grader"
TEACHER = "teaching_session"
GRADER_ATTEMPTS = 3  # the grader agent's correction attempts: this many invalid outputs fail the grade
INVALID_GRADE = json.dumps({"score": CANARY, "criterion_results": [], "outcome": "CORRECT", "confidence": 0.9})
UNCERTAIN_EXPLANATION = "Your answer could not be graded reliably, so it does not count yet."


def assert_clean(response) -> None:
    text = response.text
    assert CANARY not in text, (response.request.url, text[:2000])
    assert "input_value" not in text and "validation error" not in text.lower(), (response.request.url, text[:2000])
    assert "uncertainty_reason" not in text and "AgentOutputError" not in text, (response.request.url, text[:2000])
    assert "semantic grader" not in text.lower(), (response.request.url, text[:2000])


def assert_internal_reason(reason: str | None) -> None:
    """The vulnerable text exists internally: the raw grader error, canary and validation detail included."""
    assert reason and CANARY in reason and "input_value" in reason, reason


def canary_item(lesson_task: str) -> dict:
    """The assessment demo's free-text item, with the canary as its expected answer."""
    entry = next(e for e in fixture()["items"] if e["item"]["assessment_item_id"] == "why_preterite")
    return {"item": {**entry["item"], "assessment_item_id": "why_preterite_canary", "lesson_id": lesson_task,
                     "expected_answer": CANARY, "acceptable_answers": []},
            "rubric": entry["rubric"]}


def test_attempt_and_grade_routes_carry_no_answer_key(teaching_env) -> None:
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner, roles=(AUTHOR,)) as client:
        created = client.post("/assessment-items", json=canary_item(env.lesson_task))
        assert created.status_code == 201, created.text
        item_id = created.json()["assessment_item_id"]

        env.llm.inject(GRADER, *[INVALID_GRADE] * GRADER_ATTEMPTS)
        calls = env.llm.calls[GRADER]
        r = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": "porque sí", "attempt_id": "canary-1"})
        assert env.llm.calls[GRADER] - calls == GRADER_ATTEMPTS  # every injected output was consumed

        # The vulnerable path ran: the internal grade and event keep the raw grader error, canary included.
        stored = c.assessment_service.stored_grade("canary-1")
        assert_internal_reason(stored.uncertainty_reason)
        events = [e for e in c.task_service.events(env.lesson_task) if e.type == "assessment.uncertain"
                  and e.data.get("attempt_id") == "canary-1"]
        assert len(events) == 1
        assert_internal_reason(events[0].data["reason"])

        # The learner gets an UNCERTAIN grade with fixed feedback, and nothing of the grader's error.
        assert r.status_code == 201, r.text
        assert_clean(r)
        body = r.json()
        grade = body["grade"]
        assert grade["outcome"] == "UNCERTAIN" and "uncertainty_reason" not in grade
        assert grade["feedback"]["explanation"] == UNCERTAIN_EXPLANATION and grade["feedback"]["next_hint"]
        assert body["attempt"]["retry_recommended"] and not body["attempt"]["mastery_updated"]

        got = client.get("/assessment-attempts/canary-1/grade")
        assert got.status_code == 200, got.text
        assert_clean(got)
        assert got.json() == grade
        assert_clean(client.get("/assessment-attempts/canary-1"))
        assert_clean(client.get(f"/assessment-items/{item_id}/attempts", params={"learner_id": learner}))

        # Resubmitting what the learner was shown does not earn a CORRECT grade (the grader is back to normal).
        shown = f"{grade['feedback']['explanation']} {grade['feedback']['next_hint']}"
        retry = client.post(f"/assessment-items/{item_id}/attempts",
                            json={"learner_id": learner, "answer": shown, "attempt_id": "canary-2"})
        assert retry.status_code == 201, retry.text
        assert_clean(retry)
        assert retry.json()["grade"]["outcome"] != "CORRECT"
        assert retry.json()["attempt"]["attempt_number"] == 2

    # The key itself is what the leak exposed: it grades CORRECT by exact match, with no model call.
    calls = env.llm.calls[GRADER]
    direct = asyncio.run(c.assessment_service.submit_attempt(
        item_id, AttemptSubmission(learner_id=learner, answer=CANARY, attempt_id="canary-3")))
    assert direct.grade.outcome.value == "CORRECT" and direct.grade.grader_type.value == "EXACT"
    assert env.llm.calls[GRADER] == calls


def free_text_teacher(request):
    """The mock teacher, except that every question it asks is free text whose expected answer is the canary, so the
    session's answers go to the semantic grader."""
    out = teach(request)
    if out.get("question"):
        out["question"] = {"kind": "free_text", "prompt": out["question"]["prompt"], "choices": [],
                           "expected_answer": CANARY, "accepted_answers": []}
        out["expected_response_type"] = "free_text"
        out["response"] = out["response"].split(" Options: ")[0]
    return out


def test_teaching_answer_assessment_carries_no_answer_key(teaching_env) -> None:
    env = teaching_env
    c = env.container
    env.llm._responders = {**env.llm._responders, TEACHER: free_text_teacher}
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner) as client:
        started = client.post(f"/lessons/{env.lesson_task}/teaching-session", json={"idempotency_key": "grader-leak"})
        assert started.status_code in (200, 201), started.text
        assert_clean(started)
        sid = started.json()["session_id"]
        question = client.get(f"/teaching-sessions/{sid}").json()["state"]["waiting_question"]
        assert question["kind"] == "free_text", question

        env.llm.inject(GRADER, *[INVALID_GRADE] * GRADER_ATTEMPTS)
        calls = env.llm.calls[GRADER]
        r = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": "porque sí", "client_turn_id": "g1"})
        assert env.llm.calls[GRADER] - calls == GRADER_ATTEMPTS  # the semantic grader was invoked and failed

        assert r.status_code == 200, r.text
        body = r.json()
        attempt_id = body["assessment"]["attempt_id"]
        assert_internal_reason(c.assessment_service.stored_grade(attempt_id).uncertainty_reason)
        assert_clean(r)
        assert body["answer_outcome"] == "UNCERTAIN" and body["correct"] is None
        assert body["assessment"]["outcome"] == "UNCERTAIN" and "uncertainty_reason" not in body["assessment"]
        assert body["assessment"]["feedback"]["explanation"] == UNCERTAIN_EXPLANATION

        # Existing semantics: an UNCERTAIN answer leaves the same question open for another answer.
        view = client.get(f"/teaching-sessions/{sid}")
        assert_clean(view)
        v = view.json()
        assert v["status"] == "WAITING_FOR_LEARNER" and v["state"]["waiting_question"]["prompt"] == question["prompt"]
        assert_clean(client.get(f"/assessment-attempts/{attempt_id}/grade"))

        # The replay of the same request serves the same clean body.
        replay = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": "porque sí", "client_turn_id": "g1"})
        assert replay.status_code == 200 and replay.json()["replayed"]
        assert_clean(replay)

        # Answering with what the learner was shown does not earn a CORRECT grade.
        shown = " ".join(filter(None, [body["assessment"]["feedback"]["explanation"],
                                       body["assessment"]["feedback"]["next_hint"]]))
        again = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": shown, "client_turn_id": "g2"})
        assert again.status_code == 200, again.text
        assert_clean(again)
        assert again.json()["answer_outcome"] != "CORRECT"
