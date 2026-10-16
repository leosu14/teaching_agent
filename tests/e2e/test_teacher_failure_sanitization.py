"""PR17 regression: a teacher failure must not leak an answer key to learners.

The teacher's structured output can fail validation, and the validation error quotes the model's output
(`input_value='...'`), which can be an answer key (the teacher is given the open question's key). That text stays in the
internal agent events; the teaching routes' 503 body and the learning cycle's `failure.message` carry fixed text only.
The failure is injected explicitly: every attempt of the teacher's turn returns a canary in a mistyped field."""

from __future__ import annotations

import asyncio
import json

from app.teaching.errors import TeacherUnavailable
from tests.auth import client_for
from tests.learning_cycle_fixtures import LEARNER as CYCLE_LEARNER
from tests.teaching_fixtures import KEY

CANARY = "SECRET_ANSWER_CANARY_7F31"
SESSION_MESSAGE = "The teacher could not produce a response. You can retry the session."
CYCLE_MESSAGE = "The teacher could not produce a response. You can retry the learning cycle."
AGENT = "teaching_session"
TURN_ATTEMPTS = 3  # the teacher agent's correction attempts for one turn: this many invalid outputs fail the turn
# A HINT turn that echoes an answer key in a field with the wrong type (`citations` is a list).
INVALID_TURN = json.dumps({"action": "HINT", "concept_id": "es.past_contrast", "difficulty": 1, "response": "Hint.",
                           "expected_response_type": "free_text", "citations": CANARY})


def internal_text(events) -> str:
    return json.dumps([e.model_dump(mode="json") for e in events])


def assert_clean(response) -> None:
    text = response.text
    assert CANARY not in text, (response.request.url, text[:2000])
    assert "input_value" not in text and "validation error" not in text.lower(), (response.request.url, text[:2000])
    assert "TeacherTurnFailed" not in text and "ValidationError" not in text


def test_teacher_unavailable_text_is_fixed() -> None:
    try:
        try:
            raise ValueError(f"1 validation error for TeacherTurnOutput citations input_value='{CANARY}'")
        except ValueError as cause:
            raise TeacherUnavailable() from cause
    except TeacherUnavailable as exc:
        assert str(exc) == SESSION_MESSAGE and exc.args == (SESSION_MESSAGE,)
        assert CANARY in str(exc.__cause__)  # the diagnostic cause stays on the chain


def test_teaching_route_503_carries_no_answer_key(teaching_env) -> None:
    env = teaching_env
    c = env.container
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner) as client:
        sid = client.post(f"/lessons/{env.lesson_task}/teaching-session", json={"idempotency_key": "leak"}).json()[
            "session_id"]
        question = client.get(f"/teaching-sessions/{sid}").json()["state"]["waiting_question"]
        wrong = next(a["incorrect"] for cue, a in KEY.items() if cue in question["prompt"])
        task_id = asyncio.run(c.teaching_service.get(sid)).task_id
        events_before = len(c.task_service.events(task_id))

        env.llm.inject(AGENT, *[INVALID_TURN] * TURN_ATTEMPTS)
        calls = env.llm.calls[AGENT]
        r = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": wrong, "client_turn_id": "w1"})
        assert env.llm.calls[AGENT] - calls == TURN_ATTEMPTS  # every injected output was consumed

        # The vulnerable path ran: the internal events quote the invalid output, canary included.
        internal = c.task_service.events(task_id)[events_before:]
        failed = [e for e in internal if e.type == "agent.validation_failed"]
        assert failed and all(CANARY in json.dumps(e.data) and "input_value" in json.dumps(e.data) for e in failed)
        # The learner gets the status and the fixed message, nothing else.
        assert r.status_code == 503
        assert_clean(r)
        assert r.json() == {"error": "TeacherUnavailable", "detail": SESSION_MESSAGE}

        # Existing semantics: the answer is stored, the reply is owed, and resuming produces it.
        view = client.get(f"/teaching-sessions/{sid}")
        assert_clean(view)
        v = view.json()
        assert v["status"] == "ACTIVE" and v["next_action"]["kind"] == "teacher_reply_pending"
        assert [t["turn_type"] for t in v["turns"]] == ["EXPLANATION", "QUESTION", "LEARNER_ANSWER"]
        resumed = client.post(f"/teaching-sessions/{sid}/resume")
        assert resumed.status_code == 200, resumed.text
        assert_clean(resumed)
        after = client.get(f"/teaching-sessions/{sid}").json()
        assert after["status"] == "WAITING_FOR_LEARNER" and after["turns"][-1]["turn_type"] == "HINT"


def test_cycle_teacher_failure_message_carries_no_answer_key(cycle_env_at) -> None:
    env = cycle_env_at("session")  # the cycle's teaching session waits on its first question
    c = env.container
    with client_for(c, CYCLE_LEARNER) as client:
        [listed] = client.get(f"/learners/{CYCLE_LEARNER}/learning-cycles").json()
        cid = listed["cycle_id"]
        cycle = client.get(f"/learning-cycles/{cid}").json()
        assert cycle["prompt"]["kind"] == "SESSION_QUESTION", cycle
        [sid] = [s["child_id"] for s in cycle["steps"] if s["kind"] == "TEACHING_SESSION"]
        task_id = asyncio.run(c.teaching_service.get(sid)).task_id
        events_before = len(c.task_service.events(task_id))

        # The answer's teacher turn fails, and so does the cycle's own retry of it (two turns' worth of attempts).
        env.llm.inject(AGENT, *[INVALID_TURN] * (2 * TURN_ATTEMPTS))
        calls = env.llm.calls[AGENT]
        r = client.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "r1", "answer": "hola"})
        assert env.llm.calls[AGENT] - calls == 2 * TURN_ATTEMPTS

        internal = c.task_service.events(task_id)[events_before:]
        assert CANARY in internal_text(internal) and "input_value" in internal_text(internal)
        assert r.status_code == 200, r.text
        assert_clean(r)
        body = r.json()
        assert body["status"] == "BLOCKED"
        failure = body["failure"]
        assert failure["message"] == CYCLE_MESSAGE
        assert failure["kind"] == "PROVIDER" and failure["retryable"] is True and failure["step_id"]

        got = client.get(f"/learning-cycles/{cid}")
        assert_clean(got)
        assert got.json()["status"] == "BLOCKED" and got.json()["failure"] == failure
        listing = client.get(f"/learners/{CYCLE_LEARNER}/learning-cycles")
        assert_clean(listing)
        [same] = [x for x in listing.json() if x["cycle_id"] == cid]
        assert same["failure"] == failure
        assert_clean(client.get(f"/learning-cycles/{cid}/events"))

        # Existing semantics: the failure is retryable and resuming the cycle continues the session.
        resumed = client.post(f"/learning-cycles/{cid}/resume")
        assert resumed.status_code == 200, resumed.text
        assert_clean(resumed)
        assert resumed.json()["status"] != "BLOCKED" and resumed.json()["failure"] is None
