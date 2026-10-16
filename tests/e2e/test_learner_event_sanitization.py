"""PR17 regression: answer keys must not leak through learner-facing events or task errors.

A structured-output validation failure quotes the model's invalid output (`input_value='...'`), and that output can be
an answer key. The internal event log and the internal task record keep that text for diagnostics; the learner-facing
`/tasks/{id}/events`, `/learning-cycles/{id}/events` and task views must not carry it. The failure is injected
explicitly (the mock LLM never produces invalid output on its own): the agent's first outputs put a canary answer key
in an answer-key field with the wrong type."""

from __future__ import annotations

import json

import pytest

from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.runtime.failures import FailureCategory
from app.schemas.events import (
    LEARNER_EVENT_MESSAGES,
    Event,
    EventStatus,
    EventType,
    LearnerEvent,
)
from app.schemas.task import (
    GENERIC_TASK_ERROR,
    TASK_ERROR_MESSAGES,
    TaskError,
    TaskErrorView,
)
from app.services.container import build_container
from tests.auth import client_for
from tests.conftest import ANSWERS, EVALUATION_ANSWERS
from tests.conftest import LEARNER as DEMO
from tests.learning_cycle_fixtures import LEARNER as CYCLE_LEARNER

CANARY = "SECRET_ANSWER_CANARY_7F31"
# Invalid outputs that exhaust every correction attempt (3) of every node attempt (2), and no more.
EXHAUST = 6
# `EvaluationStep.assessment.questions[0]`: a short-answer question whose `accepted_answers` is a string, not a list.
INVALID_EVALUATION = json.dumps({"stage": "assess", "assessment": {
    "title": "t", "level": "A2", "objectives": ["o"], "total_points": 1, "rationale": "r", "questions": [{
        "question_id": "q1", "concept_id": "c1", "objective": "o", "kind": "short_answer", "prompt": "p",
        "difficulty": 0.5, "expected_answer": CANARY, "accepted_answers": CANARY}]}})
# `DiagnosticStep.items[0]`: the same slip in a diagnostic item.
INVALID_DIAGNOSTIC = json.dumps({"status": "ask", "concepts": [], "items": [{
    "question": {"question_id": "q1", "concept_id": "c1", "prompt": "p", "difficulty": 0.5},
    "expected_answer": CANARY, "accepted_answers": CANARY}]})
# Event data keys that are diagnostics only: none may reach a learner.
DIAGNOSTIC_KEYS = {"data", "error", "attempt", "uri", "provider", "model", "cost_usd", "input_tokens",
                   "output_tokens", "agent_id", "tool", "category"}
LEARNER_EVENT_FIELDS = {"event_id", "type", "status", "message", "task_id", "node_id", "at"}
TASK_ERROR_FIELDS = {"category", "stage", "node_id", "message", "at"}


def keys_in(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_in(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_in(v)}
    return set()


def assert_learner_safe(response) -> None:
    """No canary, no host path, no provider/model/cost metadata in a learner-visible body."""
    assert response.status_code in (200, 201), response.text
    text = response.text
    assert CANARY not in text, (response.request.url, text[:2000])
    assert "input_value" not in text and "file://" not in text and "/data/objects" not in text
    assert "mock-" not in text  # mock model ids
    assert not keys_in(response.json()) & (DIAGNOSTIC_KEYS - {"category"}), response.request.url


def assert_learner_events(response) -> list[dict]:
    assert_learner_safe(response)
    events = response.json()
    assert events and all(set(e) == LEARNER_EVENT_FIELDS for e in events)
    for e in events:
        assert e["message"] == LEARNER_EVENT_MESSAGES[EventStatus(e["status"])]
    return events


def internal_text(events: list[Event]) -> str:
    return json.dumps([e.model_dump(mode="json") for e in events])


@pytest.fixture
def world(settings):
    llm = MockLLMProvider(default_responders())
    c = build_container(settings, llm_providers={"mock": llm})
    lid = DEMO["learner_id"]
    with client_for(c, lid) as client:
        client.put(f"/learners/{lid}", json=DEMO["profile"])
        task = client.post("/tasks", json={"request": DEMO["request"], "learner_id": lid}).json()
        while task["status"] == "WAITING":
            sheet = task["waiting"]["prompt"]
            key = ANSWERS["rounds"][sheet["round_number"] - 1]
            task = client.post(f"/tasks/{task['task_id']}/answers", json={"answers": [
                {"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]}).json()
        assert task["status"] == "COMPLETED", task
        yield c, llm, client, task["task_id"]
    c.close()


def evaluation_answers(sheet: dict) -> dict:
    by_concept = EVALUATION_ANSWERS["answers"]
    return {"answers": [{"question_id": q["question_id"], "answer": by_concept[q["concept_id"]][q["kind"]]}
                        for q in sheet["questions"]]}


def test_validation_failure_with_answer_key_is_not_served_to_learners(world) -> None:
    c, llm, client, lesson = world
    llm.inject("learner_evaluation", INVALID_EVALUATION)  # one slip: the correction attempt succeeds

    created = client.post(f"/tasks/{lesson}/evaluation")
    assert_learner_safe(created)
    evaluation = created.json()
    tid = evaluation["task_id"]
    assert evaluation["status"] == "WAITING" and evaluation["errors"] == []

    # The internal log keeps the full diagnostic, canary included (the trusted path).
    internal = c.task_service.events(tid)
    failed = [e for e in internal if e.type == EventType.AGENT_VALIDATION_FAILED]
    assert len(failed) == 1 and CANARY in failed[0].data["error"] and "input_value" in failed[0].data["error"]

    events = assert_learner_events(client.get(f"/tasks/{tid}/events"))
    assert [e["event_id"] for e in events] == [e.event_id for e in internal]  # the same events, sanitized
    assert [e["type"] for e in events] == [e.type for e in internal]
    retry = next(e for e in events if e["type"] == EventType.AGENT_VALIDATION_FAILED)
    assert retry["status"] == "retrying" and retry["task_id"] == tid
    assert events[-1]["type"] == EventType.TASK_WAITING and events[-1]["status"] == "waiting"
    assert_learner_safe(client.get(f"/tasks/{tid}"))

    # Workflow behavior is unchanged: the evaluation completes as it would without the slip.
    done = client.post(f"/tasks/{tid}/answers", json=evaluation_answers(evaluation["waiting"]["prompt"]))
    assert_learner_safe(done)
    assert done.json()["status"] == "COMPLETED" and done.json()["result"]["score"] is not None
    events = assert_learner_events(client.get(f"/tasks/{tid}/events"))
    assert events[-1]["type"] == EventType.TASK_COMPLETED and events[-1]["status"] == "completed"
    assert CANARY in internal_text(c.task_service.events(tid))


def test_exhausted_validation_retries_give_learners_a_generic_task_error(world) -> None:
    c, llm, client, lesson = world
    llm.inject("learner_evaluation", *[INVALID_EVALUATION] * EXHAUST)

    created = client.post(f"/tasks/{lesson}/evaluation")
    assert_learner_safe(created)
    tid = created.json()["task_id"]
    for body in (created.json(), client.get(f"/tasks/{tid}").json()):
        assert body["status"] == "FAILED"
        [error] = body["errors"]
        assert set(error) == TASK_ERROR_FIELDS
        assert error["category"] == FailureCategory.EVALUATION.value
        assert error["message"] == TASK_ERROR_MESSAGES[FailureCategory.EVALUATION.value]
        assert error["node_id"] == "assess"
    assert_learner_safe(client.get(f"/tasks/{tid}"))

    # The internal record keeps the raw error for diagnostics.
    [internal_error] = c.task_service.get(tid).errors
    assert CANARY in internal_error.message and "no valid output" in internal_error.message
    assert CANARY in internal_text(c.task_service.events(tid))

    events = assert_learner_events(client.get(f"/tasks/{tid}/events"))
    assert events[-1]["type"] == EventType.TASK_FAILED and events[-1]["status"] == "failed"
    assert {e["type"] for e in events} >= {EventType.AGENT_VALIDATION_FAILED, EventType.NODE_RETRY}

    # A failed evaluation stays resumable once the model behaves.
    assert llm.calls["learner_evaluation"] == EXHAUST
    resumed = client.post(f"/tasks/{tid}/resume")
    assert_learner_safe(resumed)
    assert resumed.json()["status"] == "WAITING"


def test_cycle_and_child_task_responses_carry_no_answer_key(cycle_env_at) -> None:
    env = cycle_env_at("base")
    c = env.container
    env.llm.inject("knowledge_diagnostic", *[INVALID_DIAGNOSTIC] * EXHAUST)
    with client_for(c, CYCLE_LEARNER) as client:
        started = client.post(f"/learners/{CYCLE_LEARNER}/learning-cycles", json={"idempotency_key": "leak"})
        assert_learner_safe(started)
        cycle = started.json()
        cid = cycle["cycle_id"]
        assert cycle["status"] in ("BLOCKED", "FAILED"), cycle
        [child] = [s["child_id"] for s in cycle["steps"] if s.get("child_id")]

        internal_child = c.task_service.get(child)
        assert CANARY in internal_child.errors[-1].message
        assert_learner_safe(client.get(f"/tasks/{child}"))
        assert client.get(f"/tasks/{child}").json()["errors"][-1]["message"] in TASK_ERROR_MESSAGES.values()
        assert_learner_events(client.get(f"/tasks/{child}/events"))

        # Anything a cycle event carries in its payload stays internal: inject a diagnostic payload directly.
        injected = c.events.emit(EventType.LEARNING_CYCLE_STEP_COMPLETED, task_id=cid, node_id="learning_cycle",
                                 agent_id="learner_evaluation", tool="store", error=f"input_value='{CANARY}'",
                                 uri="file:///srv/data/objects/sha256/ab", provider="openai", model="gpt-x",
                                 cost_usd=0.42, input_tokens=10)
        assert CANARY in internal_text(c.learning_cycle_service.events(cid))
        assert_learner_safe(client.get(f"/learning-cycles/{cid}"))
        events = assert_learner_events(client.get(f"/learning-cycles/{cid}/events"))
        assert events[0]["type"] == EventType.LEARNING_CYCLE_STARTED and events[0]["status"] == "started"
        [mine] = [e for e in events if e["event_id"] == injected.event_id]
        assert mine == {"event_id": injected.event_id, "type": EventType.LEARNING_CYCLE_STEP_COMPLETED,
                        "status": "completed", "message": LEARNER_EVENT_MESSAGES[EventStatus.COMPLETED],
                        "task_id": cid, "node_id": "learning_cycle", "at": mine["at"]}


def test_learner_event_is_built_from_approved_fields_only() -> None:
    raw = Event(event_id="e1", type=EventType.TASK_FAILED, task_id="t1", node_id="evaluate", agent_id="a", tool="x",
                data={"error": CANARY, "uri": "file:///x", "provider": "p", "model": "m", "cost_usd": 1.0})
    view = LearnerEvent.of(raw)
    assert set(view.model_dump()) == LEARNER_EVENT_FIELDS
    assert view.status == EventStatus.FAILED and view.message == LEARNER_EVENT_MESSAGES[EventStatus.FAILED]
    assert CANARY not in view.model_dump_json()
    # An event type outside the known catalogue is not echoed either.
    unknown = LearnerEvent.of(Event(event_id="e2", type=f"custom.{CANARY}"))
    assert unknown.type == "other" and CANARY not in unknown.model_dump_json()
    assert LearnerEvent.of(Event(event_id="e3", type=EventType.NODE_RETRY)).status == EventStatus.RETRYING
    assert LearnerEvent.of(Event(event_id="e4", type=EventType.TASK_WAITING)).status == EventStatus.WAITING


def test_task_error_view_never_copies_the_internal_message() -> None:
    assert set(TASK_ERROR_MESSAGES) == {c.value for c in FailureCategory}  # every category has a fixed message
    raw = TaskError(kind="workflow", message=f"StructuredOutputError: input_value='{CANARY}'", node_id="evaluate",
                    category=FailureCategory.EVALUATION.value, stage=FailureCategory.EVALUATION.value)
    view = TaskErrorView.of(raw)
    assert set(view.model_dump()) == TASK_ERROR_FIELDS
    assert CANARY not in view.model_dump_json()
    assert view.message == TASK_ERROR_MESSAGES[FailureCategory.EVALUATION.value]
    odd = TaskErrorView.of(TaskError(kind="workflow", message=CANARY, category=CANARY, stage=CANARY))
    assert odd.category is None and odd.stage is None and odd.message == GENERIC_TASK_ERROR
    assert CANARY not in odd.model_dump_json()
