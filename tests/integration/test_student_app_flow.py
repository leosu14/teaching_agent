"""The student web app's API path (MVP item 6), over HTTP exactly as app/api/web/app.js calls it, on mock providers.

The learner's journey: who am I -> profile -> goal (on a course's scale) -> plan -> next action -> cycle -> diagnostic
-> teaching session -> lesson reading -> outcome -> progress. And the lesson reading's guarantee: its text never states
an answer the learner is graded on with the lesson, nor the answer of a question open for them now."""

from __future__ import annotations

import json

import pytest

from app.agents.visual.briefing import answer_keys
from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.learning_cycle import LearnerPrompt
from app.services.container import build_container
from app.services.lessons import load_lesson
from app.utils.text import reveals
from tests.auth import client_for, settle
from tests.learning_cycle_fixtures import LEARNER, ScriptedLearner


def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def stated(reading: dict, answers: list[str]) -> list[str]:
    """The reading's strings (ids aside) that state one of the answers."""
    texts = strings({k: v for k, v in reading.items() if k not in ("lesson_id", "task_id")})
    return [t for t in texts if reveals(t, answers)]


def lesson_of(env) -> str:
    [cycle] = env.container.learning_cycle_service.repository.for_learner(LEARNER)
    return next(s.child_id for s in cycle.steps if s.kind.value == "LESSON")


def answer_cycle(client, view: dict, learner: ScriptedLearner, *, until: str | None = None) -> dict:
    for _ in range(40):
        if view["status"] != "WAITING" or view["prompt"]["kind"] == until:
            return view
        response = learner.respond(LearnerPrompt.model_validate(view["prompt"]))
        r = client.post(f"/learning-cycles/{view['cycle_id']}/responses",
                        json=response.model_dump(mode="json", exclude_none=True))
        assert r.status_code == 200, r.text
        view = r.json()
    raise AssertionError("the cycle did not finish")


def test_the_app_journey_over_http(cycle_env_at) -> None:
    env = cycle_env_at("base")
    with client_for(env.container, LEARNER) as c:
        me = c.get("/me").json()
        assert me["learner_ids"] == [LEARNER] and me["roles"] == []
        assert c.get(f"/learners/{LEARNER}").status_code == 200
        [goal] = c.get(f"/learners/{LEARNER}/goals").json()
        assert c.get(f"/goals/{goal['goal_id']}/curriculum").status_code == 404  # the app offers "Plan my path"
        assert c.get(f"/learners/{LEARNER}/next-action").json()["action"] == "WAIT"
        planned = c.post(f"/goals/{goal['goal_id']}/curriculum")
        assert planned.status_code == 201, planned.text
        action = c.get(f"/learners/{LEARNER}/next-action").json()
        assert action["action"] == "LEARN"

        view = c.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "web-1"}).json()
        assert view["prompt"]["kind"] == "DIAGNOSTIC_QUESTIONS"
        lesson_task = view["steps"][0]["child_id"]
        busy = c.get(f"/lessons/{lesson_task}/reading")
        assert busy.status_code == 409  # the lesson is not written before its diagnostic is answered

        learner = ScriptedLearner(prefix="w")
        view = answer_cycle(c, view, learner, until="SESSION_QUESTION")
        session = next(s["child_id"] for s in view["steps"] if s["kind"] == "TEACHING_SESSION")
        assert c.get(f"/teaching-sessions/{session}").json()["turns"]  # the chat transcript the app shows
        reading = c.get(f"/lessons/{lesson_task}/reading")
        assert reading.status_code == 200, reading.text
        assert reading.json()["sections"] and reading.json()["title"]

        done = answer_cycle(c, view, learner)
        assert done["status"] == "COMPLETED" and done["outcome"]["mastery_changes"]
        progress = c.get(f"/learners/{LEARNER}/progress").json()
        assert action["concept_id"] in progress["mastered"]
        [listed] = c.get(f"/learners/{LEARNER}/learning-cycles").json()
        assert listed["status"] == "COMPLETED"


def test_the_reading_never_states_the_lessons_answer_keys(cycle_env_at) -> None:
    env = cycle_env_at("done")
    lesson = load_lesson(env.container.artifacts, env.container.task_service, lesson_of(env))
    keys = answer_keys(lesson.lesson)
    assert keys
    with client_for(env.container, LEARNER) as c:
        by_task = c.get(f"/lessons/{lesson.task.task_id}/reading").json()
        by_artifact = c.get(f"/lessons/{lesson.artifact.artifact_id}/reading").json()
    assert by_task == by_artifact  # the two ways a lesson is addressed
    assert stated(by_task, keys) == []
    assert not by_task["question_open"]
    assert {"exercises", "check_questions"}.isdisjoint(by_task) and "narration" not in by_task["sections"][0]
    for prompt in [*(e.prompt for e in lesson.lesson.exercises), *(q.prompt for q in lesson.lesson.check_questions)]:
        assert prompt not in json.dumps(by_task, ensure_ascii=False)


def test_an_open_session_question_is_held_back_while_it_is_open(cycle_env_at) -> None:
    env = cycle_env_at("session")
    c = env.container
    [session] = [s for s in c.teaching_service.sessions(LEARNER) if not s.terminal]
    question = session.state.pending_question
    assert question is not None
    answers = [question.expected_answer, *question.accepted_answers]
    lesson = load_lesson(c.artifacts, c.task_service, lesson_of(env))
    full = " ".join([lesson.lesson.introduction, lesson.lesson.summary,
                     *(s.explanation for s in lesson.lesson.sections), *(e for s in lesson.lesson.sections
                                                                         for e in s.examples)])
    # The scenario only proves something if the lesson text itself states the open answer somewhere.
    planted = reveals(full, answers)
    with client_for(c, LEARNER) as client:
        reading = client.get(f"/lessons/{lesson.task.task_id}/reading").json()
    assert stated(reading, answers) == []
    if planted:
        assert reading["question_open"] and reading["withheld"] >= 1


def test_held_back_text_comes_back_once_the_question_is_closed(cycle_env_at) -> None:
    """The open question's answer is withheld only while it is open; the lesson's own keys stay withheld."""
    env = cycle_env_at("session")
    c = env.container
    service = c.lesson_reading
    task = lesson_of(env)
    word = "anoche"  # stated in the lesson's explanation, and the answer of the question we open
    lesson = load_lesson(c.artifacts, c.task_service, task)
    assert any(reveals(s.explanation, [word]) for s in lesson.lesson.sections)

    open_now = service.open_answers
    service.open_answers = lambda learner_id: [*open_now(learner_id), word]
    try:
        held = service.reading(task)
    finally:
        service.open_answers = open_now
    shown = service.reading(task)
    assert stated(held.model_dump(), [word]) == [] and held.question_open
    assert stated(shown.model_dump(), [word]) and held.withheld > shown.withheld
    assert stated(shown.model_dump(), answer_keys(lesson.lesson)) == []


def test_an_item_registered_on_the_lesson_is_held_back(cycle_env_at) -> None:
    env = cycle_env_at("done")
    task = lesson_of(env)
    with client_for(env.container, LEARNER, roles=("author",)) as c:
        before = c.get(f"/lessons/{task}/reading").json()
        assert stated(before, ["anoche"])
        item = {"item": {"assessment_item_id": "reading-item", "lesson_id": task, "concept_id": "es.preterite",
                         "prompt": "When did it happen? Last ___ (night)", "expected_answer": "anoche",
                         "response_type": "SHORT_TEXT", "language": "es"}}
        assert c.post("/assessment-items", json=item).status_code == 201
        after = c.get(f"/lessons/{task}/reading").json()
    assert stated(after, ["anoche"]) == [] and after["withheld"] > before["withheld"]
    assert not after["question_open"]  # an item is always held back, not only while open


def test_a_waiting_evaluation_is_held_back(cycle_env_at) -> None:
    env = cycle_env_at("done")
    task = lesson_of(env)
    with client_for(env.container, LEARNER) as c:
        evaluation = settle(c, c.post(f"/tasks/{task}/evaluation"))
        assert evaluation["status"] == "WAITING"
        internal = env.container.task_service.get(evaluation["task_id"])
        from app.services.lesson_reading import answer_strings

        answers = answer_strings(internal.model_dump(mode="json", include={"workflow"}))
        assert answers  # the evaluation's keys exist internally
        reading = c.get(f"/lessons/{task}/reading").json()
    assert stated(reading, answers) == []


def test_courses_and_a_goal_on_the_course_scale(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", log_json=False, courses="spanish-a1-b1",
                        visual_failure_policy="continue")
    container = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    try:
        with client_for(container, "course-web") as c:
            [course] = c.get("/courses").json()
            assert course["course_id"] == "spanish-a1-b1" and course["levels"] == ["A1", "A2", "B1"]
            assert "probe" not in json.dumps(course).replace('"probes"', "")  # counts only, no probe or answer
            assert c.put("/learners/course-web", json={"display_name": "Ana",
                                                      "preferences": {"session_minutes": 30}}).status_code == 200
            body = {"title": "Reach A2 Spanish", "domain": course["subject"], "target_level": "A2",
                    "idempotency_key": "goal-web-1", "metadata": {"framework_id": course["framework"]}}
            goal = c.post("/learners/course-web/goals", json=body)
            assert goal.status_code == 201, goal.text
            assert c.post("/learners/course-web/goals", json=body).json()["goal_id"] == goal.json()["goal_id"]
            assert c.post(f"/goals/{goal.json()['goal_id']}/curriculum").status_code == 201
            assert c.get("/learners/course-web/next-action").json()["action"] == "LEARN"
    finally:
        container.close()


def test_courses_are_empty_on_the_fixture_knowledge_base(container) -> None:
    with client_for(container, "nobody") as c:
        assert c.get("/courses").json() == []


@pytest.mark.parametrize("path", ["/app/", "/app/app.js", "/app/app.css"])
def test_the_app_is_served_by_the_api_itself(container, path) -> None:
    with client_for(container, "nobody") as c:
        r = c.get(path, headers={"Authorization": ""})
    assert r.status_code == 200 and r.headers["content-security-policy"].startswith("default-src 'none'")
