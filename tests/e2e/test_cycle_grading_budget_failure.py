"""Issue #22 regression: a budget refusal while grading a session answer inside a learning cycle.

A free-text answer is graded by the semantic grader, charged to the attempt's budget, the session's budget and the daily
caps. When one of them refuses, the answer is not stored. The cycle used to swallow the refusal and wait on the same
question again with no message, for every later answer too. It now follows the teacher-reply policy (F4): the
session's or the attempt's own budget FAILS the cycle; a learner or global daily cap BLOCKS it until the UTC day
resets, after which the same question is asked again and the answer (even under the same client_response_id) is
graded. No refusal sends a provider request, and no body names a budget scope, id, limit or amount. A grader outage is
not a budget failure: the answer is stored as UNCERTAIN and the cycle goes on."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.curriculum import cycle as engine
from app.observability.spend import global_day_scope, learner_day_scope
from app.schemas.assessment import ResponseType
from app.teaching import grading
from tests.auth import client_for
from tests.e2e.test_cycle_budget_failure import (
    BUDGET_MESSAGE,
    CAP,
    DAILY_MESSAGE,
    DAY,
    Clock,
    _assert_no_internals,
    _session_cycle,
)
from tests.learning_cycle_fixtures import LEARNER

GRADER = "semantic_grader"
GRADER_ATTEMPTS = 3  # the grader agent's correction attempts: this many invalid outputs leave no valid grade
ANSWER = "I think it is probably something about the weather today"


@pytest.fixture(autouse=True)
def free_text_questions(monkeypatch) -> None:
    """The fixture's session question is short-answer, graded without a model call. Grade it as free text (the kind
    the semantic grader sees), as the issue's reproduction did."""
    monkeypatch.setitem(grading.RESPONSE_TYPES, "short_answer", ResponseType.FREE_TEXT)


def _answer(client, cid: str, rid: str, answer: str = ANSWER):
    r = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": rid, "answer": answer})
    assert r.status_code == 200, r.text
    return r


def _learner_answers(env, sid: str) -> list[str]:
    return [t.content for t in env.container.teaching_service.turns(sid) if t.speaker.value == "LEARNER"]


@pytest.mark.parametrize("setting", ["budget_session_max_llm_requests", "budget_attempt_max_llm_requests"])
def test_own_budget_refused_at_grading_fails_the_cycle(cycle_env_at, setting) -> None:
    env = cycle_env_at("session", **{setting: 0})
    with client_for(env.container, LEARNER) as client:
        cid, sid = _session_cycle(client)
        requests, graded = len(env.llm.requests), env.llm.calls[GRADER]
        r = _answer(client, cid, "r1")
        body = r.json()
        assert body["status"] == "FAILED", body  # was WAITING on the same question, with no message
        kind, retryable = engine.classify_task_failure("BudgetExceededError")
        failure = body["failure"]
        assert (failure["kind"], failure["retryable"], failure["message"], failure["category"]) == \
            (kind.value, retryable, BUDGET_MESSAGE, "BudgetExceededError")
        assert failure["step_id"] and body["prompt"] is None
        assert (len(env.llm.requests), env.llm.calls[GRADER]) == (requests, graded)  # refused before it was sent
        assert ANSWER not in _learner_answers(env, sid)  # nothing was graded or stored
        _assert_no_internals(r.text, sid)
        _assert_no_internals(client.get(f"/learning-cycles/{cid}").text, sid)
        _assert_no_internals(client.get(f"/learning-cycles/{cid}/events").text, sid)

        # Nothing to retry: the cycle cannot be resumed or answered, and trying sends no provider request.
        assert client.post(f"/learning-cycles/{cid}/resume").status_code == 409
        again = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "r2", "answer": ANSWER})
        assert again.status_code == 409, again.text
        assert len(env.llm.requests) == requests


@pytest.mark.parametrize(("setting", "scope_key"), [
    ("budget_learner_daily_max_llm_requests", lambda: learner_day_scope(LEARNER, DAY)),
    ("budget_global_daily_max_llm_requests", lambda: global_day_scope(DAY)),
])
def test_daily_cap_refused_at_grading_blocks_until_the_day_resets(cycle_env_at, setting, scope_key) -> None:
    clock = Clock(DAY)
    env = cycle_env_at("session", clock=clock, **{setting: CAP})
    env.container.teaching_service._clock = clock  # the grading account's UTC day (the container passes no clock)
    env.container.assessment_service._clock = clock
    assert env.container.spend.seed(scope_key(), {"llm_requests": CAP}, now=DAY)  # today's cap is used up
    with client_for(env.container, LEARNER) as client:
        cid, sid = _session_cycle(client)
        question = client.get(f"/learning-cycles/{cid}").json()["prompt"]
        requests, graded = len(env.llm.requests), env.llm.calls[GRADER]
        r = _answer(client, cid, "r1")
        body = r.json()
        assert body["status"] == "BLOCKED", body  # was WAITING on the same question, with no message
        failure = body["failure"]
        assert (failure["kind"], failure["retryable"], failure["message"], failure["category"]) == \
            ("WORKFLOW", True, DAILY_MESSAGE, None)
        assert len(env.llm.requests) == requests
        assert ANSWER not in _learner_answers(env, sid)
        _assert_no_internals(r.text, sid)
        _assert_no_internals(client.get(f"/learning-cycles/{cid}/events").text, sid)

        # The same day: resuming asks the same question again (nothing is owed, so nothing is sent); answering it is
        # refused again at reservation and blocks again.
        same_day = client.post(f"/learning-cycles/{cid}/resume")
        assert same_day.status_code == 200, same_day.text
        assert same_day.json()["status"] == "WAITING" and same_day.json()["prompt"] == question
        blocked = _answer(client, cid, "r1")  # the refused response's id was not used up
        assert blocked.json()["status"] == "BLOCKED" and blocked.json()["failure"]["message"] == DAILY_MESSAGE
        assert len(env.llm.requests) == requests

        # The next UTC day: the same question, and the same answer is graded this time.
        clock.t = DAY + timedelta(hours=2)
        resumed = client.post(f"/learning-cycles/{cid}/resume")
        assert resumed.status_code == 200, resumed.text
        view = resumed.json()
        assert view["status"] == "WAITING" and view["failure"] is None and view["prompt"] == question, view
        assert [s["child_id"] for s in view["steps"] if s["kind"] == "TEACHING_SESSION"] == [sid]
        done = _answer(client, cid, "r1")
        assert done.json()["status"] in ("WAITING", "RUNNING", "COMPLETED"), done.json()
        assert done.json()["failure"] is None
        assert env.llm.calls[GRADER] > graded  # graded now
        assert ANSWER in _learner_answers(env, sid)


def test_grader_outage_is_not_a_budget_failure(cycle_env_at) -> None:
    """Every grader attempt returns invalid output: the answer is stored as UNCERTAIN and the cycle goes on."""
    env = cycle_env_at("session")
    with client_for(env.container, LEARNER) as client:
        cid, sid = _session_cycle(client)
        graded = env.llm.calls[GRADER]
        env.llm.inject(GRADER, *["not a grade"] * GRADER_ATTEMPTS)
        r = _answer(client, cid, "r1")
        body = r.json()
        assert body["status"] not in ("FAILED", "BLOCKED") and body["failure"] is None, body
        assert env.llm.calls[GRADER] == graded + GRADER_ATTEMPTS  # the grader was really called
        assert ANSWER in _learner_answers(env, sid)
        assert DAILY_MESSAGE not in r.text and BUDGET_MESSAGE not in r.text
