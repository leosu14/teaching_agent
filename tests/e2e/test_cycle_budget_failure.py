"""F4 regression: a learning cycle whose teaching session has used up its budget fails for good.

Retrying cannot succeed while the same budget stays used up, so the cycle must not be BLOCKED with the teacher's
"you can retry" text (a provider outage): it FAILS, classified like a child task's exhausted budget, with fixed text
that names no budget scope, id, limit or amount. A daily cap (the learner's or the global one) has room again the next
UTC day, so it BLOCKS the cycle instead, and resuming after the reset continues the same session. A genuine teacher
outage stays retryable."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.curriculum import cycle as engine
from app.observability.spend import global_day_scope, learner_day_scope
from app.schemas.learning_cycle import FailureKind
from tests.auth import client_for
from tests.e2e.test_teacher_failure_sanitization import AGENT, INVALID_TURN, TURN_ATTEMPTS
from tests.learning_cycle_fixtures import LEARNER

BUDGET_MESSAGE = "The spending limit for this work has been reached. The learning cycle cannot continue."
DAILY_MESSAGE = ("Today's spending limit has been reached. You can resume the learning cycle once the daily limit "
                 "resets.")
TEACHER_MESSAGE = "The teacher could not produce a response. You can retry the learning cycle."
# What a learner-facing body must never carry about the budget that refused the work.
INTERNALS = ("session:", "learner:", "global:", "budget exceeded", "MAX_LLM", "max_llm", "(limit ", ", used ",
             "Raise the limit", "Try again later", "TeacherUnavailable", "BudgetExhausted")


def _session_cycle(client) -> tuple[str, str]:
    [listed] = client.get(f"/learners/{LEARNER}/learning-cycles").json()
    cid = listed["cycle_id"]
    cycle = client.get(f"/learning-cycles/{cid}").json()
    assert cycle["prompt"]["kind"] == "SESSION_QUESTION", cycle
    [sid] = [s["child_id"] for s in cycle["steps"] if s["kind"] == "TEACHING_SESSION"]
    return cid, sid


def _assert_no_internals(text: str, session_id: str) -> None:
    for word in INTERNALS:
        assert word not in text, (word, text[:2000])
    assert f"session:{session_id}" not in text  # the budget's scope key (the step's child id is public)


def test_exhausted_session_budget_fails_the_cycle_without_retry_advice(cycle_env_at) -> None:
    # The session's teacher turns may make no further request: the reply owed after the answer is refused.
    env = cycle_env_at("session", budget_session_max_llm_requests=0)
    with client_for(env.container, LEARNER) as client:
        cid, sid = _session_cycle(client)
        r = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "r1", "answer": "hola"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "FAILED", body
        failure = body["failure"]
        assert failure["message"] == BUDGET_MESSAGE
        assert failure["retryable"] is False and failure["step_id"]
        # The same classification a child task's exhausted budget gets.
        kind, retryable = engine.classify_task_failure("BudgetExceededError")
        assert (failure["kind"], failure["retryable"], failure["category"]) == \
            (kind.value, retryable, "BudgetExceededError")
        _assert_no_internals(r.text, sid)

        got = client.get(f"/learning-cycles/{cid}")
        assert got.json()["status"] == "FAILED" and got.json()["failure"] == failure
        _assert_no_internals(got.text, sid)
        [same] = [x for x in client.get(f"/learners/{LEARNER}/learning-cycles").json() if x["cycle_id"] == cid]
        assert same["failure"] == failure
        _assert_no_internals(client.get(f"/learning-cycles/{cid}/events").text, sid)

        # Nothing to retry: a failed cycle cannot be resumed, and trying sends no provider request.
        requests = len(env.llm.requests)
        assert client.post(f"/learning-cycles/{cid}/resume").status_code == 409
        assert len(env.llm.requests) == requests


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


DAY = datetime(2030, 1, 1, 23, 0, tzinfo=timezone.utc)
CAP = 50


@pytest.mark.parametrize(("setting", "scope_key"), [
    ("budget_learner_daily_max_llm_requests", lambda: learner_day_scope(LEARNER, DAY)),
    ("budget_global_daily_max_llm_requests", lambda: global_day_scope(DAY)),
])
def test_exhausted_daily_cap_blocks_the_cycle_until_the_day_resets(cycle_env_at, setting, scope_key) -> None:
    clock = Clock(DAY)
    env = cycle_env_at("session", clock=clock, **{setting: CAP})
    env.container.teaching_service._clock = clock  # the session account's UTC day (the container passes no clock)
    assert env.container.spend.seed(scope_key(), {"llm_requests": CAP}, now=DAY)  # today's cap is used up
    with client_for(env.container, LEARNER) as client:
        cid, sid = _session_cycle(client)
        requests = len(env.llm.requests)
        r = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "r1", "answer": "hola"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "BLOCKED", body
        failure = body["failure"]
        assert (failure["kind"], failure["retryable"], failure["message"], failure["category"]) == \
            ("WORKFLOW", True, DAILY_MESSAGE, None)
        assert len(env.llm.requests) == requests  # refused before it was sent
        _assert_no_internals(r.text, sid)
        _assert_no_internals(client.get(f"/learning-cycles/{cid}").text, sid)
        _assert_no_internals(client.get(f"/learning-cycles/{cid}/events").text, sid)

        # The same day: resuming is refused again by the cap, before any provider request.
        again = client.post(f"/learning-cycles/{cid}/resume")
        assert again.status_code == 200, again.text
        assert again.json()["status"] == "BLOCKED" and again.json()["failure"]["message"] == DAILY_MESSAGE
        assert len(env.llm.requests) == requests

        # The next UTC day: the cap has room, and the same cycle continues the same session.
        clock.t = DAY + timedelta(hours=2)
        resumed = client.post(f"/learning-cycles/{cid}/resume")
        assert resumed.status_code == 200, resumed.text
        view = resumed.json()
        assert view["status"] == "WAITING" and view["failure"] is None, view
        assert view["prompt"]["kind"] == "SESSION_QUESTION"
        assert [s["child_id"] for s in view["steps"] if s["kind"] == "TEACHING_SESSION"] == [sid]
        assert len(env.llm.requests) > requests  # the owed reply was produced


def test_teacher_outage_stays_retryable(cycle_env_at) -> None:
    """A genuine provider failure of the same owed reply: BLOCKED, retryable, and resuming continues the session."""
    env = cycle_env_at("session")
    with client_for(env.container, LEARNER) as client:
        cid, sid = _session_cycle(client)
        # The answer's teacher turn fails, and so does the cycle's own retry of it (two turns' worth of attempts).
        env.llm.inject(AGENT, *[INVALID_TURN] * (2 * TURN_ATTEMPTS))
        r = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "r1", "answer": "hola"})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "BLOCKED"
        failure = r.json()["failure"]
        assert (failure["kind"], failure["retryable"], failure["message"], failure["category"]) == \
            ("PROVIDER", True, TEACHER_MESSAGE, None)
        _assert_no_internals(r.text, sid)

        resumed = client.post(f"/learning-cycles/{cid}/resume")
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] != "BLOCKED" and resumed.json()["failure"] is None


def test_task_budget_classification_is_unchanged() -> None:
    """A child task's exhausted budget keeps its classification (the session's now matches it), and a task's
    provider failure stays retryable."""
    assert engine.classify_task_failure("BudgetExceededError") == (FailureKind.WORKFLOW, False)
    assert engine.classify_task_failure("ProviderError") == (FailureKind.PROVIDER, True)
