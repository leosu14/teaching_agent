"""PR17 regression (NEW-1, N7): an open teaching question's stored item must not be an answer oracle.

After the learner's first answer the open question is stored as an ASSESSMENT_ITEM artifact on the learner's own
lesson task (and its turn as a TEACHING_TURN artifact). Every field of the item except the key is public to the
learner, so:
- NEW-1: `GET /tasks/{id}/artifacts` served each artifact's `content_hash` and `size_bytes`. The size gave the key's
  length and the hash verified guessed keys offline, while the question was open.
- N7: `POST /assessment-items/{id}/attempts` graded guesses on that same item outside the session.
The served artifact now carries no digest or byte length (its own or in its metadata), and the attempts route answers a
session's own question item with the uniform 404. The internal artifact keeps its hash and size. The mock teacher here
makes `SECRET_ANSWER_CANARY_7F31` the open question's key."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from app.providers.llm.mock_responders import teach
from app.schemas.assessment import AssessmentItem
from app.schemas.learner import stable_id
from app.schemas.teaching import TeachingConfig
from tests.auth import client_for

CANARY = "SECRET_ANSWER_CANARY_7F31"
AGENT = "teaching_session"
RESPONSE_TYPE = {"short_answer": "SHORT_TEXT", "multiple_choice": "MULTIPLE_CHOICE"}
CHOICES = ["WRONG_CHOICE_ALPHA", CANARY, "WRONG_CHOICE_OMEGA"]
QUESTIONS = {
    "multiple_choice": {"kind": "multiple_choice", "prompt": "Pick the canary.", "choices": CHOICES,
                        "expected_answer": CANARY, "accepted_answers": []},
    "short_answer": {"kind": "short_answer", "prompt": "Type the canary.", "choices": [],
                     "expected_answer": CANARY, "accepted_answers": []},
}
WRONG = {"multiple_choice": "WRONG_CHOICE_ALPHA", "short_answer": "NOT_THE_CANARY"}
# guesses the learner could test offline: the key, the other choices, and strings of every nearby length
GUESSES = [CANARY, *CHOICES, "SECRET_ANSWER_CANARY_0000", *("X" * n for n in range(1, 40))]
DIGEST_OR_LENGTH = {"content_hash", "size_bytes", "checksum", "hash", "digest", "sha256", "bytes", "file_size",
                    "content_length", "byte_length", "key", "object_key", "uri"}
HEX_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")


def teacher_with(question: dict):
    def respond(request):
        out = teach(request)
        if out.get("question"):
            out["question"] = question
            out["expected_response_type"] = "multiple_choice" if question["kind"] == "multiple_choice" else "free_text"
            out["response"] = question["prompt"] + (f" Options: {' / '.join(CHOICES)}" if question["choices"] else "")
        return out
    return respond


def rebuilt(view: dict, q: dict, guess: str, language: str) -> bytes:
    """The stored item, rebuilt from public data only (session view and waiting question) plus a guessed key."""
    sid = view["session_id"]
    item = AssessmentItem(
        assessment_item_id=stable_id("aitem", sid, q["turn_id"]), lesson_id=view["lesson_id"],
        objective_id=view["objective_id"], concept_id=view["state"]["concept_id"], prompt=q["prompt"],
        expected_answer=guess, acceptable_answers=[], response_type=RESPONSE_TYPE[q["kind"]],
        difficulty=TeachingConfig().evidence_difficulty(q["difficulty"]), language=language,
        choices=q["choices"] if q["kind"] == "multiple_choice" else [],
        metadata={"session_id": sid, "question_turn_id": q["turn_id"]})
    return json.dumps(item.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str).encode("utf-8")


def walk(value, path: str = "$"):
    """Every (path, key, value) in a JSON document, recursively."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield f"{path}.{k}", k, v
            yield from walk(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")


def open_question_after_one_wrong_answer(env, client, kind: str) -> tuple[str, dict, dict]:
    sid = client.post(f"/lessons/{env.lesson_task}/teaching-session",
                      json={"idempotency_key": f"oracle-{kind}"}).json()["session_id"]
    q = client.get(f"/teaching-sessions/{sid}").json()["state"]["waiting_question"]
    assert q["kind"] == kind and q["prompt"] == QUESTIONS[kind]["prompt"]
    r = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": WRONG[kind], "client_turn_id": "w1"})
    assert r.status_code == 200 and r.json()["answer_outcome"] == "INCORRECT", r.text[:400]
    view = client.get(f"/teaching-sessions/{sid}").json()
    assert view["status"] == "WAITING_FOR_LEARNER" and view["state"]["waiting_question"]["turn_id"] == q["turn_id"]
    return sid, view, view["state"]["waiting_question"]


@pytest.mark.parametrize("kind", ["multiple_choice", "short_answer"])
def test_open_question_artifact_is_not_an_offline_answer_oracle(teaching_env, kind: str) -> None:
    env = teaching_env
    c = env.container
    env.llm._responders = {**env.llm._responders, AGENT: teacher_with(QUESTIONS[kind])}
    learner = c.task_service.get(env.lesson_task).learner_id
    with client_for(c, learner) as client:
        sid, view, q = open_question_after_one_wrong_answer(env, client, kind)
        item_id = stable_id("aitem", sid, q["turn_id"])
        name = f"assessment_item.{item_id}"

        # the internal artifact keeps its key, hash and size, and its bytes still verify (storage integrity)
        internal = {a.name: a for a in c.task_service.artifacts(env.lesson_task)}
        stored = internal[name]
        content = c.artifacts.read_object(stored.uri)
        assert CANARY in content.decode("utf-8")
        assert stored.content_hash == hashlib.sha256(content).hexdigest() and stored.size_bytes == len(content)
        assert c.artifacts.verify(stored) is None
        # and the offline attack is real: the public data rebuilds the stored bytes exactly when the guess is the key
        language = next(lang for lang in ("en", "es", "fr", "de") if rebuilt(view, q, CANARY, lang) == content)

        response = client.get(f"/tasks/{env.lesson_task}/artifacts")
        assert response.status_code == 200
        body = response.json()
        text = response.text

        # the artifact is still listed, with its identity, to its owner
        served = next(a for a in body if a["name"] == name)
        assert served["artifact_id"] == stored.artifact_id and served["type"] == "ASSESSMENT_ITEM"
        assert {a["artifact_id"] for a in body} == {a.artifact_id for a in internal.values()}

        # no digest or byte length anywhere in the response, under any name, at any depth
        assert CANARY not in text and CANARY.lower() not in text.lower()
        for path, key, value in walk(body):
            assert key not in DIGEST_OR_LENGTH and not key.endswith(("_checksum", "_hash", "_digest")), path
            if isinstance(value, str):
                assert not HEX_DIGEST.search(value), (path, value)
        sizes = {a.size_bytes for a in internal.values()}
        assert not [p for p, _, v in walk(body) if isinstance(v, int) and not isinstance(v, bool) and v in sizes]
        hashes = {a.content_hash for a in internal.values()}
        assert not [h for h in hashes if h in text]

        # no guess, the key included, can be checked against anything served
        for guess in (CHOICES if kind == "multiple_choice" else GUESSES):  # an MC key is one of its choices
            candidate = rebuilt(view, q, guess, language)
            assert hashlib.sha256(candidate).hexdigest() not in text
            assert not [p for p, _, v in walk(served) if v == len(candidate)], guess

        # N7: the attempts API refuses the session's own item with the uniform 404, for every guess
        missing = client.post("/assessment-items/aitem_does_not_exist/attempts",
                              json={"learner_id": learner, "answer": CANARY, "attempt_id": "probe-missing"})
        assert missing.status_code == 404
        for i, guess in enumerate(CHOICES):
            r = client.post(f"/assessment-items/{item_id}/attempts",
                            json={"learner_id": learner, "answer": guess, "attempt_id": f"guess-{i}"})
            assert (r.status_code, r.json()) == (404, missing.json()), r.text[:400]
        attempts = c.assessment_service.attempts(item_id, learner)
        assert [a.learner_answer for a in attempts] == [WRONG[kind]]  # only the session's own attempt

        # the question is still open, the session saw one wrong answer, and the key still grades CORRECT there
        after = client.get(f"/teaching-sessions/{sid}").json()
        assert after["state"]["waiting_question"]["turn_id"] == q["turn_id"]
        assert after["state"]["incorrect_answers"] == 1
        r = client.post(f"/teaching-sessions/{sid}/answers", json={"answer": CANARY, "client_turn_id": "w2"})
        assert r.json()["answer_outcome"] == "CORRECT", r.text[:400]

        # internal metadata is unchanged by serving it
        again = {a.name: a for a in c.task_service.artifacts(env.lesson_task)}[name]
        assert (again.content_hash, again.size_bytes, again.metadata) == (stored.content_hash, stored.size_bytes,
                                                                          stored.metadata)
