"""PR17 security regression: authentication, learner ownership and response hardening over HTTP.

Learner A's world is the learning-cycle fixture's learner mid-cycle (goal, curriculum, lesson task, teaching session),
plus an evaluation task, an assessment item and an attempt created over HTTP. Learner B is a real learner with their
own valid credentials. Every protected route is exercised anonymously, with bad credentials, as B, and as A."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.auth import AUTHOR
from app.schemas.learner import LearnerProfileInput
from app.schemas.learning_cycle import LearnerPrompt
from tests.auth import TestAuth, client_for
from tests.learning_cycle_fixtures import CONCEPT, LEARNER, ScriptedLearner

INTRUDER = "intruder-learner"
SENTINEL = "SENTINEL-ANSWER-KEY-41f7"  # the registered item's answer key: no learner-facing response may carry it
# Keys that only internal records carry: answer keys, grading internals, workflow state, provider records, host paths.
FORBIDDEN_KEYS = {"expected_answer", "accepted_answers", "acceptable_answers", "answer_key", "correct_answer",
                  "reference_answer", "known_errors", "workflow", "provider_requests", "node_states", "uri"}
PUBLIC = {("GET", "/health")}
NOT_FOUND = {"error": "NotFound", "detail": "not found"}


def keys_in(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_in(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_in(v)}
    return set()


def assert_no_internals(response, *, events: bool = False) -> None:
    """No answer key or internal grading state in a learner-facing response. Event payloads describe pipeline steps
    (an image's "expected" size, an artifact's stored location), so only answer-key names are checked there."""
    assert SENTINEL not in response.text
    found = keys_in(response.json()) & FORBIDDEN_KEYS
    if events:
        found -= {"uri"}
    assert not found, (response.request.url, found)


@pytest.fixture
def world(cycle_env_at):
    """Learner A mid-cycle with every kind of resource; B a separate learner. Yields (env, ids, auth)."""
    env = cycle_env_at("session")
    c = env.container
    c.learner_service.upsert(INTRUDER, LearnerProfileInput(display_name="B"))
    [cycle] = c.learning_cycle_service.repository.for_learner(LEARNER)
    children = {s.kind.value: s.child_id for s in cycle.steps}
    lesson_task, session = children["LESSON"], children["TEACHING_SESSION"]
    auth = TestAuth()
    with client_for(c, LEARNER, roles=(AUTHOR,), auth=auth) as a:
        evaluation = a.post(f"/tasks/{lesson_task}/evaluation").json()["task_id"]
        item = {"item": {"assessment_item_id": "sec-item", "lesson_id": lesson_task, "concept_id": CONCEPT,
                         "prompt": "Ayer yo ___ (hablar) con María.", "expected_answer": SENTINEL,
                         "response_type": "SHORT_TEXT", "language": "es"}}
        assert a.post("/assessment-items", json=item).status_code == 201
        attempt = a.post("/assessment-items/sec-item/attempts",
                         json={"learner_id": LEARNER, "answer": "hablaba", "attempt_id": "sec-attempt"})
        assert attempt.status_code == 201, attempt.text
    ids = {"learner": LEARNER, "goal": c.curriculum_service.goals(LEARNER)[0].goal_id, "cycle": cycle.cycle_id,
           "task": lesson_task, "evaluation": evaluation, "session": session, "item": "sec-item",
           "attempt": "sec-attempt", "lesson": c.artifacts.find(lesson_task, "lesson").artifact_id}
    yield env, ids, auth


def routes(ids: dict) -> list[tuple[str, str, dict | None]]:
    """Every protected route, addressed at learner A's resources. Bodies are valid so that only authorization can
    stop a request."""
    lid, gid, tid, eid, sid, cid = ids["learner"], ids["goal"], ids["task"], ids["evaluation"], ids["session"], \
        ids["cycle"]
    return [
        ("GET", "/providers", None), ("GET", "/agents", None), ("GET", "/tools", None),
        ("PUT", f"/learners/{lid}", {"display_name": "hijacked"}),
        ("GET", f"/learners/{lid}", None),
        ("GET", f"/learners/{lid}/progress", None),
        ("POST", f"/learners/{lid}/assessment", {"task_id": eid, "answers": [{"question_id": "q", "answer": "a"}]}),
        ("POST", f"/learners/{lid}/goals", {"title": "Other goal", "domain": "spanish", "target_level": "A2"}),
        ("GET", f"/learners/{lid}/goals", None),
        ("GET", f"/learners/{lid}/next-action", None),
        ("GET", f"/goals/{gid}", None),
        ("PATCH", f"/goals/{gid}", {"target_level": "B2"}),
        ("POST", f"/goals/{gid}/curriculum", None),
        ("GET", f"/goals/{gid}/curriculum", None),
        ("GET", f"/goals/{gid}/curriculum/versions", None),
        ("POST", "/tasks", {"request": "Teach me Spanish", "learner_id": lid}),
        ("GET", f"/tasks/{tid}", None),
        ("POST", f"/tasks/{tid}/evaluation", None),
        ("POST", f"/tasks/{eid}/answers", {"answers": [{"question_id": "q", "answer": "a"}]}),
        ("POST", f"/tasks/{eid}/pause", None),
        ("POST", f"/tasks/{eid}/resume", None),
        ("POST", f"/tasks/{eid}/cancel", None),
        ("GET", f"/tasks/{tid}/artifacts", None),
        ("GET", f"/tasks/{tid}/events", None),
        ("POST", f"/lessons/{ids['lesson']}/teaching-session", {"idempotency_key": "intrusion"}),
        ("GET", f"/teaching-sessions/{sid}", None),
        ("POST", f"/teaching-sessions/{sid}/answers", {"answer": "x", "client_turn_id": "intrusion"}),
        ("POST", f"/teaching-sessions/{sid}/pause", None),
        ("POST", f"/teaching-sessions/{sid}/resume", None),
        ("POST", f"/teaching-sessions/{sid}/cancel", None),
        ("POST", "/assessment-items", {"item": {"assessment_item_id": "sec-item-2", "lesson_id": tid,
                                                "concept_id": CONCEPT, "prompt": "p", "expected_answer": "k",
                                                "response_type": "SHORT_TEXT"}}),
        ("GET", f"/assessment-items/{ids['item']}/attempts?learner_id={lid}", None),
        ("POST", f"/assessment-items/{ids['item']}/attempts", {"learner_id": lid, "answer": "x"}),
        ("GET", f"/assessment-attempts/{ids['attempt']}", None),
        ("GET", f"/assessment-attempts/{ids['attempt']}/grade", None),
        ("POST", f"/learners/{lid}/learning-cycles", {"idempotency_key": "intrusion"}),
        ("GET", f"/learners/{lid}/learning-cycles", None),
        ("GET", f"/learning-cycles/{cid}", None),
        ("POST", f"/learning-cycles/{cid}/responses", {"client_response_id": "intrusion", "answer": "x"}),
        ("POST", f"/learning-cycles/{cid}/resume", None),
        ("POST", f"/learning-cycles/{cid}/cancel", None),
        ("GET", f"/learning-cycles/{cid}/events", None),
        ("GET", f"/learning-cycles/{cid}/artifacts", None),
    ]


def template(path: str, ids: dict) -> str:
    """`/tasks/task_123/events?x=1` -> `/tasks/{task_id}/events`, to match the app's route table."""
    path = path.split("?")[0]
    names = {ids["learner"]: "learner_id", ids["goal"]: "goal_id", ids["task"]: "task_id",
             ids["evaluation"]: "task_id", ids["session"]: "session_id", ids["cycle"]: "cycle_id",
             ids["item"]: "item_id", ids["attempt"]: "attempt_id", ids["lesson"]: "lesson_id"}
    return "/".join("{" + names[part] + "}" if part in names else part for part in path.split("/"))


def send(client: TestClient, method: str, path: str, body: dict | None, headers: dict | None = None):
    return client.request(method, path, json=body, headers=headers)


def snapshot(client: TestClient, ids: dict) -> dict:
    """Learner A's observable state, to prove that refused requests changed nothing."""
    reads = [f"/learners/{ids['learner']}", f"/learners/{ids['learner']}/goals", f"/goals/{ids['goal']}",
             f"/goals/{ids['goal']}/curriculum/versions", f"/tasks/{ids['task']}", f"/tasks/{ids['evaluation']}",
             f"/teaching-sessions/{ids['session']}", f"/learners/{ids['learner']}/learning-cycles",
             f"/assessment-items/{ids['item']}/attempts?learner_id={ids['learner']}"]
    out = {}
    for path in reads:
        r = client.get(path)
        assert r.status_code == 200, (path, r.text)
        out[path] = r.json()
    return out


def test_every_protected_route_is_in_the_matrix(world) -> None:
    env, ids, auth = world
    app = create_app(env.container, authenticator=auth.authenticator)
    served = {(m.upper(), path) for path, ops in app.openapi()["paths"].items() for m in ops}
    covered = {(m, template(p, ids)) for m, p, _ in routes(ids)}
    assert served - PUBLIC == covered  # a new route must be added to the matrix (and so be checked) to pass


def test_unauthenticated_and_invalid_credentials_are_rejected_everywhere(world) -> None:
    env, ids, auth = world
    with TestClient(create_app(env.container, authenticator=auth.authenticator)) as anonymous:
        assert anonymous.get("/health").status_code == 200  # the one public route
        bad_headers = [{"Authorization": "Bearer not-a-real-token-at-all"}, {"Authorization": "Basic Zm9vOmJhcg=="},
                       {"Authorization": "Bearer "}, {"X-Auth-Subject": LEARNER}]
        for method, path, body in routes(ids):
            r = send(anonymous, method, path, body)
            assert r.status_code == 401, (method, path, r.status_code)
            assert r.json() == {"error": "Unauthenticated", "detail": "authentication required"}
            assert r.headers["www-authenticate"] == "Bearer"
            for headers in bad_headers:
                assert send(anonymous, method, path, body, headers).status_code == 401, (method, path, headers)


def test_learner_b_cannot_reach_any_of_learner_a_resources(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a, \
            client_for(env.container, INTRUDER, roles=(AUTHOR,), auth=auth) as b:
        before = snapshot(a, ids)
        for method, path, body in routes(ids):
            r = send(b, method, path, body)
            if path in ("/providers", "/agents", "/tools"):
                assert r.status_code == 200  # not learner data: any authenticated principal
                continue
            assert r.status_code == 404, (method, path, r.status_code, r.text)
            assert r.json() == NOT_FOUND, (method, path)  # the same answer as for an id that does not exist
        assert snapshot(a, ids) == before  # nothing B sent changed anything of A's


def test_not_yours_is_indistinguishable_from_missing(world) -> None:
    env, ids, auth = world
    with client_for(env.container, INTRUDER, auth=auth) as b:
        for path in (f"/tasks/{ids['task']}", f"/goals/{ids['goal']}", f"/teaching-sessions/{ids['session']}",
                     f"/learning-cycles/{ids['cycle']}", f"/assessment-attempts/{ids['attempt']}",
                     f"/learners/{ids['learner']}/progress"):
            missing = path.rsplit("/", 1)[0] + "/does_not_exist" if "progress" not in path \
                else "/learners/does_not_exist/progress"
            theirs, nothing = b.get(path), b.get(missing)
            assert (theirs.status_code, theirs.json()) == (nothing.status_code, nothing.json()) == (404, NOT_FOUND)


def test_learner_a_reaches_their_own_resources_without_internal_state(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        for method, path, _ in routes(ids):
            if method != "GET":
                continue
            r = a.get(path)
            assert r.status_code == 200, (path, r.text)
            if path not in ("/agents", "/tools"):  # the catalog describes schemas, not anyone's data
                assert_no_internals(r, events=path.endswith("/events"))
        # the internal records still carry what grading needs: only the responses changed
        internal = env.container.task_service.get(ids["evaluation"]).model_dump_json()
        assert "expected_answer" in internal
        assert env.container.assessment_service.item("sec-item").expected_answer == SENTINEL


def test_the_author_role_is_needed_to_register_items(world) -> None:
    env, ids, auth = world
    body = {"item": {"assessment_item_id": "sec-item-3", "lesson_id": ids["task"], "concept_id": CONCEPT,
                     "prompt": "p", "expected_answer": "k", "response_type": "SHORT_TEXT"}}
    with client_for(env.container, LEARNER, auth=auth) as a:
        r = a.post("/assessment-items", json=body)
        assert r.status_code == 403 and r.json()["error"] == "Forbidden"
        author = auth.headers(LEARNER, user_id="teacher", roles=(AUTHOR,))
        assert a.post("/assessment-items", json=body, headers=author).status_code == 201
        # an author outside the lesson's learner scope cannot register on it, nor take over an existing item id
        outsider = auth.headers(INTRUDER, user_id="other-teacher", roles=(AUTHOR,))
        assert a.post("/assessment-items", json=body, headers=outsider).json() == NOT_FOUND


def test_get_task_never_exposes_the_answer_key(container) -> None:
    """The audit's probe (probe_answer_key_leak.py), adapted: a WAITING diagnostic and the completed task."""
    from tests.conftest import ANSWERS
    from tests.conftest import LEARNER as DEMO

    lid = DEMO["learner_id"]
    with client_for(container, lid) as a:
        a.put(f"/learners/{lid}", json=DEMO["profile"])
        task = a.post("/tasks", json={"request": DEMO["request"], "learner_id": lid}).json()
        assert task["status"] == "WAITING"
        while task["status"] == "WAITING":
            fetched = a.get(f"/tasks/{task['task_id']}")
            assert "expected_answer" not in fetched.text and "accepted_answers" not in fetched.text
            assert_no_internals(fetched)
            sheet = fetched.json()["waiting"]["prompt"]
            key = ANSWERS["rounds"][sheet["round_number"] - 1]
            answers = [{"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]
            task = a.post(f"/tasks/{task['task_id']}/answers", json={"answers": answers}).json()
        done = a.get(f"/tasks/{task['task_id']}")
        assert done.json()["status"] == "COMPLETED"
        assert_no_internals(done)
        assert len(done.content) < 20_000  # the full internal record is hundreds of kilobytes
        profile = a.get(f"/learners/{lid}")
        assert_no_internals(profile)
        assert "expected" not in keys_in(profile.json())  # the stored profile keeps the keys it was graded against
        assert "expected_answer" in container.task_service.get(task["task_id"]).model_dump_json()  # still internal


def test_cross_learner_attack_end_to_end(cycle_env_at) -> None:
    """Learner A owns a goal, starts a learning cycle, reaches a lesson, a teaching session, an evaluation and a
    graded attempt. Learner B, authenticated and with their own data, attacks every one of them by id and through
    nested routes. Every attempt fails with the same 404, A's state is untouched, and A carries on."""
    env = cycle_env_at("base")
    c = env.container
    auth = TestAuth()
    with client_for(c, LEARNER, roles=(AUTHOR,), auth=auth) as a, \
            client_for(c, INTRUDER, roles=(AUTHOR,), auth=auth) as b:
        assert b.put(f"/learners/{INTRUDER}", json={"display_name": "B"}).status_code == 200

        # A: goal -> cycle -> lesson (diagnostic) -> teaching session
        [goal] = a.get(f"/learners/{LEARNER}/goals").json()
        view = a.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "a-cycle"}).json()
        assert view["status"] == "WAITING" and view["prompt"]["kind"] == "DIAGNOSTIC_QUESTIONS"
        learner = ScriptedLearner(prefix="a")
        while view["prompt"]["kind"] != "SESSION_QUESTION":
            response = learner.respond(LearnerPrompt.model_validate(view["prompt"]))
            view = a.post(f"/learning-cycles/{view['cycle_id']}/responses",
                          json=response.model_dump(mode="json", exclude_none=True)).json()
        cycle = view["cycle_id"]
        steps = {s["kind"]: s["child_id"] for s in view["steps"]}
        lesson, session = steps["LESSON"], steps["TEACHING_SESSION"]
        evaluation = a.post(f"/tasks/{lesson}/evaluation").json()
        assert evaluation["status"] == "WAITING" and "expected_answer" not in json.dumps(evaluation)
        item = {"item": {"assessment_item_id": "a-item", "lesson_id": lesson, "concept_id": CONCEPT,
                         "prompt": "Ayer yo ___ (hablar).", "expected_answer": SENTINEL, "response_type": "SHORT_TEXT",
                         "language": "es"}}
        assert a.post("/assessment-items", json=item).status_code == 201
        assert a.post("/assessment-items/a-item/attempts",
                      json={"learner_id": LEARNER, "answer": "hablaba", "attempt_id": "a-attempt"}).status_code == 201
        before = {p: a.get(p).json() for p in (
            f"/learners/{LEARNER}", f"/goals/{goal['goal_id']}", f"/learning-cycles/{cycle}", f"/tasks/{lesson}",
            f"/tasks/{evaluation['task_id']}", f"/teaching-sessions/{session}", "/assessment-attempts/a-attempt")}

        attacks = [
            # direct object-id substitution: A's ids on routes B may use
            ("GET", f"/learning-cycles/{cycle}", None),
            ("POST", f"/learning-cycles/{cycle}/responses", {"client_response_id": "b1", "answer": "fue"}),
            ("POST", f"/learning-cycles/{cycle}/cancel", None),
            ("GET", f"/tasks/{lesson}", None),
            ("GET", f"/tasks/{evaluation['task_id']}", None),
            ("POST", f"/tasks/{evaluation['task_id']}/answers", {"answers": [{"question_id": "q", "answer": "a"}]}),
            ("POST", f"/tasks/{evaluation['task_id']}/cancel", None),
            ("POST", f"/tasks/{lesson}/evaluation", None),
            ("GET", f"/teaching-sessions/{session}", None),
            ("POST", f"/teaching-sessions/{session}/answers", {"answer": "fue", "client_turn_id": "b1"}),
            ("POST", f"/teaching-sessions/{session}/cancel", None),
            ("GET", f"/goals/{goal['goal_id']}", None),
            ("PATCH", f"/goals/{goal['goal_id']}", {"target_level": "B2"}),
            ("POST", f"/goals/{goal['goal_id']}/curriculum", None),
            ("GET", "/assessment-attempts/a-attempt/grade", None),
            ("PUT", f"/learners/{LEARNER}", {"display_name": "hijacked"}),
            ("POST", f"/learners/{LEARNER}/learning-cycles", {"idempotency_key": "a-cycle"}),
            # nested attacks: B's own route (or B as the named learner) carrying A's object ids
            ("POST", f"/learners/{INTRUDER}/assessment",
             {"task_id": evaluation["task_id"], "answers": [{"question_id": "q", "answer": "a"}]}),
            ("POST", "/tasks", {"request": "Teach me", "learner_id": LEARNER}),
            ("POST", f"/lessons/{lesson}/teaching-session", {"idempotency_key": "b"}),
            ("POST", "/assessment-items/a-item/attempts", {"learner_id": INTRUDER, "answer": SENTINEL}),
            ("GET", f"/assessment-items/a-item/attempts?learner_id={INTRUDER}", None),
            ("POST", "/assessment-items", {"item": {**item["item"], "expected_answer": "rewritten"}}),
            ("POST", "/assessment-items", {"item": {**item["item"], "assessment_item_id": "b-item"}}),
        ]
        for method, path, body in attacks:
            r = b.request(method, path, json=body)
            assert (r.status_code, r.json()) == (404, NOT_FOUND), (method, path, r.status_code, r.text)
        # B's own (empty) scope still works: authorization is per resource, not a blanket refusal
        assert b.get(f"/learners/{INTRUDER}/learning-cycles").json() == []

        assert {p: a.get(p).json() for p in before} == before  # every refused request changed nothing
        assert c.assessment_service.item("a-item").expected_answer == SENTINEL

        # A carries on: their own mutations still succeed
        response = learner.respond(LearnerPrompt.model_validate(view["prompt"]))
        moved = a.post(f"/learning-cycles/{cycle}/responses", json=response.model_dump(mode="json", exclude_none=True))
        assert moved.status_code == 200, moved.text
        assert_no_internals(moved)
