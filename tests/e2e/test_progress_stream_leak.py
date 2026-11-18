"""Issue #54, the answer-leak regression sweep over every streamed frame.

The streams serve only `TaskView`, `LearningCycleView` and `LearnerEvent`, the projections the GET routes serve, so
they must pass the same oracles:
- the canary of tests/e2e/test_learner_event_sanitization.py (an answer key quoted by a validation failure, and an
  event payload full of diagnostics) appears in no frame of either stream, and every progress event has exactly the
  learner-safe fields;
- across a whole learning cycle (diagnostic, lesson generation, session, completion), read live through every POST,
  no frame carries an internal key (answer keys, workflow state, provider records, host paths), and no frame that
  shows an open question states that question's answer key."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.schemas.events import EventType
from app.utils.text import reveals
from tests import sse
from tests.auth import client_for
from tests.e2e.test_learner_event_sanitization import (
    CANARY,
    DIAGNOSTIC_KEYS,
    EXHAUST,
    INVALID_DIAGNOSTIC,
    LEARNER_EVENT_FIELDS,
    internal_text,
    keys_in,
)
from tests.e2e.test_progress_stream_api import FAST, Api, respond
from tests.e2e.test_security_api import FORBIDDEN_KEYS
from tests.learning_cycle_fixtures import LEARNER, ScriptedLearner


def assert_frames_safe(frames: list[dict], text: str) -> None:
    assert frames, "no frames"
    assert CANARY not in text
    assert "input_value" not in text and "file://" not in text and "/data/objects" not in text
    assert "mock-" not in text  # mock model ids
    data = [f["data"] for f in frames if "data" in f]
    assert not keys_in(data) & FORBIDDEN_KEYS
    assert not keys_in(data) & (DIAGNOSTIC_KEYS - {"category"})
    for f in frames:
        if f.get("event") == "progress":
            assert set(f["data"]) == {"source", "event"} and set(f["data"]["event"]) == LEARNER_EVENT_FIELDS


def read(client: TestClient, path: str) -> tuple[list[dict], str]:
    r = client.get(path)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream"), r.text
    return sse.parse(r.text), r.text


def test_a_validation_failure_quoting_an_answer_key_reaches_no_frame(cycle_env_at) -> None:
    env = cycle_env_at("base", **FAST)
    c = env.container
    env.llm.inject("knowledge_diagnostic", *[INVALID_DIAGNOSTIC] * EXHAUST)
    with client_for(c, LEARNER) as client:
        started = client.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "leak"}).json()
        cid = started["cycle_id"]
        assert started["status"] in ("BLOCKED", "FAILED"), started
        [child] = [s["child_id"] for s in started["steps"] if s.get("child_id")]
        c.events.emit(EventType.LEARNING_CYCLE_STEP_COMPLETED, task_id=cid, node_id="learning_cycle",
                      agent_id="learner_evaluation", tool="store", error=f"input_value='{CANARY}'",
                      uri="file:///srv/data/objects/sha256/ab", provider="openai", model="gpt-x", cost_usd=0.42,
                      input_tokens=10)
        assert CANARY in internal_text(c.learning_cycle_service.events(cid)) + c.task_service.get(child) \
            .model_dump_json()  # the internal records keep it
        for path in (f"/learning-cycles/{cid}/stream", f"/tasks/{child}/stream"):
            frames, text = read(client, path)
            assert frames[-1]["event"] == "end"
            assert any(f.get("event") == "progress" for f in frames)
            assert_frames_safe(frames, text)


def open_answers(container, frame: dict) -> set[str]:
    """The answer keys of the question(s) a WAITING cycle frame shows, from the internal records."""
    prompt = frame["data"].get("prompt")
    if frame["data"]["status"] != "WAITING" or not prompt:
        return set()
    if prompt["kind"] == "SESSION_QUESTION":
        session = container.teaching_service.repository.get(frame["data"]["waiting"]["child_id"])
        q = session.state.pending_question
        return {a for a in [q.expected_answer, *q.accepted_answers] if a} if q is not None and not q.choices \
            else set()
    ids = {q["question_id"] for q in prompt["questions"] if not q.get("choices")}
    internal = json.loads(container.task_service.get(frame["data"]["waiting"]["child_id"]).model_dump_json())
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if "expected_answer" in node and ids & set(json.dumps(node).split('"')):
                found.update(a for a in [node["expected_answer"], *(node.get("accepted_answers") or [])]
                             if isinstance(a, str) and a)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(internal)
    return found


def visible_text(frame: dict) -> str:
    """What the frame shows the learner, without the choices a multiple-choice question lists on purpose."""
    data = json.loads(json.dumps(frame["data"]))
    for q in (data.get("prompt") or {}).get("questions", []):
        q.pop("choices", None)
    question = (data.get("prompt") or {}).get("question") or {}
    question.pop("choices", None)
    return json.dumps(data, ensure_ascii=False)


def test_every_frame_of_a_whole_cycle_is_learner_safe(cycle_env_at) -> None:
    env = cycle_env_at("diagnostic", **FAST)
    api = Api(env.container, LEARNER)
    try:
        view = api.http.get(f"/learners/{LEARNER}/learning-cycles").json()[0]
        view = api.http.get(f"/learning-cycles/{view['cycle_id']}").json()
        cid = view["cycle_id"]
        learner = ScriptedLearner(prefix="sweep")
        frames: list[dict] = []
        raw = bytearray()
        checked = 0
        cursor = None
        for _ in range(40):
            if view["status"] != "WAITING":
                break
            stream = api.stream(f"/learning-cycles/{cid}/stream?after_version={view['version']}", cursor=cursor)
            posted = respond(api, view, learner)
            got = stream.rest()
            raw += stream.raw
            reply = posted()
            assert reply.status_code == 200, reply.text
            assert got[-1]["event"] == "end"
            # Checked now, while the question the stream ended on is still open: every frame that shows it.
            ref = (got[-1]["data"].get("waiting") or {}).get("ref")
            answers = open_answers(api.container, got[-1])
            checked += bool(answers)
            for f in got:
                if f.get("event") in ("status", "end") and (f["data"].get("waiting") or {}).get("ref") == ref:
                    assert not reveals(visible_text(f), answers), (f["data"]["prompt"], answers)
            frames += got
            cursor = got[-1]["id"]
            view = reply.json()
        assert view["status"] == "COMPLETED" and checked
        assert_frames_safe(frames, raw.decode("utf-8"))
        lesson = next(s["child_id"] for s in view["steps"] if s["kind"] == "LESSON")
        task_frames = api.stream(f"/tasks/{lesson}/stream").rest()
        assert_frames_safe(task_frames, json.dumps(task_frames))
        assert "expected_answer" in api.container.task_service.get(lesson).model_dump_json()  # internal keeps it
    finally:
        api.close()
