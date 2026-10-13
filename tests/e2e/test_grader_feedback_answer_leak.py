"""PR17 regression (N3): a semantic grader's learner-visible feedback must not state the answer key.

The semantic grader is given the item's expected and acceptable answers. Its output can be schema-valid and pass every
other rule while one of its learner-visible texts says the key ("the expected answer is ..."). Those texts are served
in `PublicGrade` on `POST /assessment-items/{id}/attempts`, `GET /assessment-attempts/{id}/grade` and the teaching
`/answers` response (and its replay), where the question can still be open, and the key then grades CORRECT by exact
match. The `answer_revealed` validation rule applies the teacher's `reveals()` check to every such text: a leaking
candidate is sent back to the grader, and if every attempt leaks the grade is UNCERTAIN with fixed feedback.

The grader here is a mock responder that writes an otherwise valid candidate (every rubric criterion scored 0,
INCORRECT, confidence 0.9) with the canary, the item's expected answer, in one learner-visible field."""

from __future__ import annotations

import json

import pytest

from app.api.auth import AUTHOR
from app.providers.llm.mock_responders import grade_semantically, teach
from app.schemas.assessment import SemanticGradingRequest
from scripts.run_assessment_demo import fixture
from tests.auth import client_for

CANARY = "SECRET_ANSWER_CANARY_7F31"
OTHER_KEY = "OTHER_ACCEPTED_ANSWER_2C94"
GRADER = "semantic_grader"
TEACHER = "teaching_session"
GRADER_ATTEMPTS = 3  # the grader agent's correction attempts: this many leaking outputs fail the grade
UNCERTAIN_EXPLANATION = "Your answer could not be graded reliably, so it does not count yet."
LEARNER_ANSWER = "porque sí"
# Every grader-written text a PublicGrade serves: the six feedback fields of the review, and the misconception type.
FIELDS = ["explanation", "next_hint", "strengths", "errors", "rationale", "misconception_description",
          "misconception_type"]


def leaking_candidate(request: SemanticGradingRequest, field: str, key: str = CANARY, *,
                      insufficient: bool = False) -> dict:
    """A candidate that passes every rule except `answer_revealed`: `key` is in `field`."""
    text = f"The expected answer is {key}."
    c = {"score": 0.0, "outcome": "INCORRECT", "confidence": 0.9, "citations": [], "misconceptions": [],
         "criterion_results": [{"criterion_id": k.criterion_id, "score": 0.0, "rationale": "not stated"}
                               for k in request.rubric.criteria],
         "feedback": {"strengths": [], "errors": [], "explanation": "", "next_hint": ""},
         "insufficient_context": insufficient}
    if field == "rationale":
        c["criterion_results"][0]["rationale"] = text
    elif field.startswith("misconception"):
        c["misconceptions"] = [{"concept_id": request.item.concept_id, "type": "habitual_reading",
                                "description": "reads the action as habitual", "confidence": 0.9}]
        c["misconceptions"][0]["type" if field == "misconception_type" else "description"] = text[:80]
    elif field in ("strengths", "errors"):
        c["feedback"][field] = [text]
    else:
        c["feedback"][field] = text
    return c


def leaking_grader(field: str, leaks: int = GRADER_ATTEMPTS, **kwargs):
    """The first `leaks` grader calls return a leaking candidate; later calls are the stock mock grader."""
    calls = {"n": 0}

    def respond(request):
        calls["n"] += 1
        if calls["n"] > leaks:
            return grade_semantically(request)
        return leaking_candidate(SemanticGradingRequest.model_validate(request.input_payload), field, **kwargs)
    return respond


def use_grader(env, responder) -> None:
    env.llm._responders = {**env.llm._responders, GRADER: responder}


def assert_clean(response, *keys: str) -> None:
    text = response.text
    for key in (CANARY, *keys):
        assert key not in text, (response.request.url, text[:2000])
    assert "answer_revealed" not in text and "uncertainty_reason" not in text, (response.request.url, text[:2000])


def mastery(env, learner: str) -> dict[str, float]:
    return {cid: c.mastery for cid, c in env.container.memory.get(learner).concepts.items()}


def assert_internal_diagnostics(env, attempt_id: str, field: str) -> None:
    """The rejection is kept internally (the grade's reason and the grader's validation events) with the rule and the
    leaking field; the learner-facing bodies above carry none of it."""
    reason = env.container.assessment_service.stored_grade(attempt_id).uncertainty_reason
    assert reason and "answer_revealed" in reason, reason
    failures = [e for e in env.container.task_service.events(env.lesson_task)
                if e.type == "agent.validation_failed" and "answer_revealed" in e.data.get("error", "")]
    assert failures and any(FIELD_PATHS[field] in e.data["error"] for e in failures), failures[-1:]


FIELD_PATHS = {"explanation": "feedback.explanation", "next_hint": "feedback.next_hint",
               "strengths": "feedback.strengths[0]", "errors": "feedback.errors[0]",
               "rationale": "].rationale", "misconception_description": "misconceptions[0].description",
               "misconception_type": "misconceptions[0].type"}


def canary_item(lesson_task: str, item_id: str, acceptable: list[str] = ()) -> dict:
    """The assessment demo's free-text item (rubric `completed`, `bounded`), with the canary as its expected answer."""
    entry = next(e for e in fixture()["items"] if e["item"]["assessment_item_id"] == "why_preterite")
    return {"item": {**entry["item"], "assessment_item_id": item_id, "lesson_id": lesson_task,
                     "expected_answer": CANARY, "acceptable_answers": list(acceptable)},
            "rubric": entry["rubric"]}


@pytest.mark.parametrize("field", FIELDS)
def test_attempt_routes_never_serve_a_leaking_grader_feedback(teaching_env, field) -> None:
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner, roles=(AUTHOR,)) as client:
        created = client.post("/assessment-items", json=canary_item(env.lesson_task, f"n3-{field}"))
        assert created.status_code == 201, created.text
        item_id = created.json()["assessment_item_id"]
        before = mastery(env, learner)

        use_grader(env, leaking_grader(field))
        calls = env.llm.calls[GRADER]
        r = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": LEARNER_ANSWER, "attempt_id": f"{field}-1"})
        assert r.status_code == 201, r.text
        assert_clean(r)
        assert env.llm.calls[GRADER] - calls == GRADER_ATTEMPTS  # every leaking candidate was rejected
        assert_internal_diagnostics(env, f"{field}-1", field)

        # Fail closed: UNCERTAIN, fixed feedback, a retry recommended, nothing counted.
        body = r.json()
        grade = body["grade"]
        assert grade["outcome"] == "UNCERTAIN" and grade["misconceptions"] == []
        assert grade["feedback"]["explanation"] == UNCERTAIN_EXPLANATION and grade["feedback"]["next_hint"]
        assert body["attempt"]["retry_recommended"] and not body["attempt"]["mastery_updated"]
        assert mastery(env, learner) == before

        got = client.get(f"/assessment-attempts/{field}-1/grade")
        assert got.status_code == 200, got.text
        assert_clean(got)
        assert got.json() == grade
        assert_clean(client.get(f"/assessment-attempts/{field}-1"))
        assert_clean(client.get(f"/assessment-items/{item_id}/attempts", params={"learner_id": learner}))

        # Resubmitting everything the learner was shown does not earn a CORRECT grade or move mastery.
        retry = client.post(f"/assessment-items/{item_id}/attempts",
                            json={"learner_id": learner, "answer": json.dumps(grade), "attempt_id": f"{field}-2"})
        assert retry.status_code == 201, retry.text
        assert_clean(retry)
        assert retry.json()["grade"]["outcome"] != "CORRECT"
        assert mastery(env, learner) == before


def test_a_leaking_candidate_is_corrected_on_retry(teaching_env) -> None:
    """The grader is told what to rewrite: a clean second candidate is graded normally."""
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner, roles=(AUTHOR,)) as client:
        item_id = client.post("/assessment-items", json=canary_item(env.lesson_task, "n3-retry")).json()[
            "assessment_item_id"]
        use_grader(env, leaking_grader("explanation", leaks=1))
        calls = env.llm.calls[GRADER]
        r = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": "Porque pasó ayer.", "attempt_id": "retry-1"})
        assert env.llm.calls[GRADER] - calls == 2
        assert r.status_code == 201, r.text
        assert_clean(r)
        assert r.json()["grade"]["outcome"] in ("INCORRECT", "PARTIAL")
        assert r.json()["grade"]["grader_type"] == "SEMANTIC"
        correction = [q for q in env.llm.requests if q.agent_id == GRADER][-1]
        assert any("answer_revealed" in m.content for m in correction.messages)


def test_insufficient_context_explanation_is_checked_too(teaching_env) -> None:
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner, roles=(AUTHOR,)) as client:
        item_id = client.post("/assessment-items", json=canary_item(env.lesson_task, "n3-ic")).json()[
            "assessment_item_id"]
        use_grader(env, leaking_grader("explanation", insufficient=True))
        r = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": LEARNER_ANSWER, "attempt_id": "ic-1"})
        assert r.status_code == 201, r.text
        assert_clean(r)
        assert r.json()["grade"]["outcome"] == "UNCERTAIN"
        assert r.json()["grade"]["feedback"]["explanation"] == UNCERTAIN_EXPLANATION


def test_every_accepted_answer_is_protected(teaching_env) -> None:
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner, roles=(AUTHOR,)) as client:
        item_id = client.post("/assessment-items", json=canary_item(env.lesson_task, "n3-acc", [OTHER_KEY])).json()[
            "assessment_item_id"]
        use_grader(env, leaking_grader("errors", key=OTHER_KEY))
        r = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": LEARNER_ANSWER, "attempt_id": "acc-1"})
        assert r.status_code == 201, r.text
        assert_clean(r, OTHER_KEY)
        assert r.json()["grade"]["outcome"] == "UNCERTAIN"


def test_quoting_the_learners_own_words_is_allowed(teaching_env) -> None:
    """An answer the learner already wrote discloses nothing: feedback may quote it back. Only that answer is exempt:
    the other accepted answer stays protected."""
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    own = f"No sé, quizá {CANARY} porque sí"  # says the key, but is not an exact match: graded semantically
    with client_for(c, learner, roles=(AUTHOR,)) as client:
        item_id = client.post("/assessment-items", json=canary_item(env.lesson_task, "n3-own", [OTHER_KEY])).json()[
            "assessment_item_id"]
        use_grader(env, leaking_grader("explanation", leaks=1))
        calls = env.llm.calls[GRADER]
        r = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": own, "attempt_id": "own-1"})
        assert env.llm.calls[GRADER] - calls == 1  # accepted first time
        assert r.status_code == 201, r.text
        grade = r.json()["grade"]
        assert grade["outcome"] == "INCORRECT" and grade["grader_type"] == "SEMANTIC"
        assert CANARY in grade["feedback"]["explanation"]

        use_grader(env, leaking_grader("explanation", key=OTHER_KEY))
        other = client.post(f"/assessment-items/{item_id}/attempts",
                            json={"learner_id": learner, "answer": own, "attempt_id": "own-2"})
        assert other.status_code == 201, other.text
        assert OTHER_KEY not in other.text
        assert other.json()["grade"]["outcome"] == "UNCERTAIN"


def test_the_stock_grader_no_longer_names_a_key_quoting_criterion(teaching_env) -> None:
    """The demo item's criterion `completed` is described as "says the action is completed", which contains its
    expected answer ("The action is completed."): the stock mock grader used to serve it in `errors` and `next_hint`
    of a partial answer, and the learner could resubmit it for CORRECT."""
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    entry = next(e for e in fixture()["items"] if e["item"]["assessment_item_id"] == "why_preterite")
    with client_for(c, learner, roles=(AUTHOR,)) as client:
        created = client.post("/assessment-items", json={"item": {**entry["item"], "lesson_id": env.lesson_task},
                                                          "rubric": entry["rubric"]})
        assert created.status_code == 201, created.text
        item_id = created.json()["assessment_item_id"]
        r = client.post(f"/assessment-items/{item_id}/attempts",
                        json={"learner_id": learner, "answer": "Porque pasó ayer.", "attempt_id": "stock-1"})
        assert r.status_code == 201, r.text
        grade = r.json()["grade"]
        assert grade["outcome"] in ("INCORRECT", "PARTIAL") and grade["grader_type"] == "SEMANTIC"
        assert "the action is completed" not in r.text.lower() and "action is finished" not in r.text.lower()


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


@pytest.mark.parametrize("field", FIELDS)
def test_teaching_answers_and_replay_never_serve_a_leaking_grader_feedback(teaching_env, field) -> None:
    env = teaching_env
    c = env.container
    env.llm._responders = {**env.llm._responders, TEACHER: free_text_teacher}
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner) as client:
        started = client.post(f"/lessons/{env.lesson_task}/teaching-session", json={"idempotency_key": f"n3-{field}"})
        assert started.status_code in (200, 201), started.text
        sid = started.json()["session_id"]
        question = client.get(f"/teaching-sessions/{sid}").json()["state"]["waiting_question"]
        assert question["kind"] == "free_text", question
        before = mastery(env, learner)

        use_grader(env, leaking_grader(field))
        calls = env.llm.calls[GRADER]
        r = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": LEARNER_ANSWER, "client_turn_id": "t1"})
        assert r.status_code == 200, r.text
        assert_clean(r)
        assert env.llm.calls[GRADER] - calls == GRADER_ATTEMPTS  # every leaking candidate was rejected
        body = r.json()
        assert_internal_diagnostics(env, body["assessment"]["attempt_id"], field)
        assert body["answer_outcome"] == "UNCERTAIN" and body["correct"] is None
        assert body["assessment"]["feedback"]["explanation"] == UNCERTAIN_EXPLANATION
        assert mastery(env, learner) == before

        # The question is still open, and nothing served while it is open states its answer.
        assert body["status"] == "WAITING_FOR_LEARNER"
        assert body["waiting_question"]["prompt"] == question["prompt"]
        view = client.get(f"/teaching-sessions/{sid}")
        assert_clean(view)
        assert view.json()["state"]["waiting_question"]["prompt"] == question["prompt"]
        assert_clean(client.get(f"/assessment-attempts/{body['assessment']['attempt_id']}/grade"))

        replay = client.post(f"/teaching-sessions/{sid}/answers",
                             json={"answer": LEARNER_ANSWER, "client_turn_id": "t1"})
        assert replay.status_code == 200 and replay.json()["replayed"]
        assert_clean(replay)

        # Answering with everything the learner was shown does not earn a CORRECT grade or move mastery.
        again = client.post(f"/teaching-sessions/{sid}/answers",
                            json={"answer": json.dumps(body["assessment"]), "client_turn_id": "t2"})
        assert again.status_code == 200, again.text
        assert_clean(again)
        assert again.json()["answer_outcome"] != "CORRECT"
        assert mastery(env, learner) == before
