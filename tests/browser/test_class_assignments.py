"""Class assignments end to end in a real browser (Important item 3, #62).

A teacher assigns a course topic with a due date to their class on the class page; the learner sees it on Today as a
goal "assigned by your teacher", with the due date, and may pause and resume it (nothing else). The teacher follows
the learner on the assignment's page, moves the due date into the past (the learner then sees it overdue, display
only), and cancels it. Every click goes through the pages' own scripts against the real API."""

from __future__ import annotations

import json

import pytest

from app.api.app import create_app
from app.api.auth import TEACHER
from tests.auth import TestAuth
from tests.browser.conftest import serve
from tests.learning_cycle_fixtures import LEARNER

TIMEOUT = 30000
PHONE = {"width": 390, "height": 844}


def token_for(auth: TestAuth, *learner_ids: str, **kw) -> str:
    return auth.headers(*learner_ids, **kw)["Authorization"].removeprefix("Bearer ")


@pytest.fixture
def world(cycle_env_at):
    from fastapi.testclient import TestClient

    env = cycle_env_at("base")
    c = env.container
    auth = TestAuth()
    app = create_app(c, authenticator=auth.authenticator)
    teacher_headers = auth.headers(user_id="teacher-web", roles=(TEACHER,))
    with TestClient(app) as client:
        made = client.post("/classes", json={"name": "Spanish B1"}, headers=teacher_headers).json()
        joined = client.post(f"/learners/{LEARNER}/classes", json={"join_code": made["join_code"]},
                             headers=auth.headers(LEARNER))
        assert joined.status_code == 201, joined.text
    served = serve(app)

    class World:
        url = served.url
        container = c
        class_id = made["class_id"]
        teacher = token_for(auth, user_id="teacher-web", roles=(TEACHER,))
        student = token_for(auth, LEARNER)

        def assign(self, body: dict) -> dict:
            with TestClient(app) as client:  # its own thread and event loop, not the browser's
                r = client.post(f"/classes/{made['class_id']}/assignments", json=body, headers=teacher_headers)
                assert r.status_code == 201, r.text
                return r.json()

    yield World()
    served.stop()


@pytest.fixture
def open_page(browser):
    contexts = []

    def opener(url: str, path: str, token: str, viewport: dict | None = None):
        context = browser.new_context(viewport=viewport) if viewport else browser.new_context()
        contexts.append(context)
        page = context.new_page()
        page.set_default_timeout(TIMEOUT)
        page.problems = []
        page.on("console", lambda m: page.problems.append(m.text)
                if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
        page.on("pageerror", lambda e: page.problems.append(str(e)))
        page.add_init_script(f"sessionStorage.setItem('ta.token', {json.dumps(token)})")
        page.goto(f"{url}{path}")
        return page

    yield opener
    for c in contexts:
        c.close()


def assigned_card(page):
    return page.locator("[data-testid=goal]", has=page.get_by_test_id("assigned-by"))


def test_a_teacher_assigns_a_topic_and_the_learner_works_on_it(world, open_page) -> None:
    teacher = open_page(world.url, f"/app/teacher.html#/class/{world.class_id}", world.teacher)
    teacher.get_by_text("No assignments yet").wait_for()
    teacher.get_by_test_id("assignment-title").fill("The preterite")
    teacher.get_by_test_id("assignment-description").fill("Finished actions in the past.")
    teacher.get_by_test_id("assignment-subject").fill("spanish")  # the fixture knowledge base lists no courses
    teacher.get_by_test_id("assignment-topics").fill("preterite")
    due = teacher.get_by_test_id("assignment-due-input").input_value()
    assert due.endswith("T23:59")  # a week ahead, at the end of the teacher's day
    teacher.get_by_test_id("create-assignment").click()
    row = teacher.get_by_test_id("assignment-row")
    row.wait_for()
    assert "The preterite" in row.inner_text() and "preterite" in row.inner_text()
    assert teacher.get_by_test_id("assignment-totals").inner_text() == "0 of 1 learner completed"

    # a topic the course does not have is refused with the API's reason
    teacher.get_by_test_id("assignment-title").fill("Klingon")
    teacher.get_by_test_id("assignment-subject").fill("spanish")
    teacher.get_by_test_id("assignment-topics").fill("no such topic")
    teacher.get_by_test_id("create-assignment").click()
    teacher.get_by_test_id("error").wait_for()
    assert "no topics" in teacher.get_by_test_id("error").inner_text()

    student = open_page(world.url, "/app/", world.student)
    card = assigned_card(student)
    card.wait_for()
    text = card.inner_text()
    assert "The preterite" in text and "Assigned by your teacher in Spanish B1" in text and "due" in text
    assert "Finished actions in the past." in text
    card.get_by_test_id("pause-goal").click()
    assigned_card(student).get_by_test_id("resume-goal").wait_for()
    assert "paused" in assigned_card(student).inner_text()
    assigned_card(student).get_by_test_id("resume-goal").click()
    assigned_card(student).get_by_test_id("pause-goal").wait_for()
    assigned_card(student).get_by_test_id("goal-priority").select_option("1")
    student.wait_for_function("() => document.querySelector('[data-testid=goal-priority]') && "
                              "document.querySelector('[data-testid=goal-priority]').value === '1'")

    teacher.get_by_test_id("assignment-link").click()
    teacher.get_by_test_id("assignment-summary").wait_for()
    learner = teacher.get_by_test_id("assignment-learner")
    assert learner.count() == 1 and "not started" in learner.inner_text()
    # the due date moves into the past: overdue, display only
    teacher.get_by_test_id("change-due-input").fill("2020-01-01T09:00")
    teacher.get_by_test_id("change-due").click()
    teacher.get_by_text("past due").wait_for()
    assert "overdue" in teacher.get_by_test_id("assignment-learner").inner_text()
    student.reload()
    assigned_card(student).get_by_test_id("overdue").wait_for()
    assert assigned_card(student).get_by_test_id("pause-goal").is_visible()  # still the learner's to work on

    teacher.once("dialog", lambda d: d.accept())
    teacher.get_by_test_id("cancel-assignment").click()
    teacher.get_by_test_id("assignment-learner").get_by_text("cancelled").wait_for()
    assert teacher.get_by_test_id("cancel-assignment").count() == 0
    student.reload()
    assigned_card(student).wait_for()
    assert "cancelled" in assigned_card(student).inner_text()
    assert assigned_card(student).get_by_test_id("pause-goal").count() == 0
    student.goto(f"{world.url}/app/#/progress")
    student.get_by_test_id("assignment").wait_for()
    assert "The preterite" in student.get_by_test_id("assignments").inner_text()
    assert teacher.problems == [] and student.problems == []  # no script error, no CSP violation


def test_the_assignment_pages_fit_a_phone(world, open_page) -> None:
    made = world.assign({"title": "The preterite", "domain": "spanish", "topics": ["preterite"],
                         "due_at": "2099-01-01T12:00:00Z"})
    for path, ready in ((f"/app/teacher.html#/class/{world.class_id}", "assignment-row"),
                        (f"/app/teacher.html#/class/{world.class_id}/assignment/{made['assignment_id']}",
                         "assignment-learner")):
        page = open_page(world.url, path, world.teacher, viewport=PHONE)
        page.get_by_test_id(ready).wait_for()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), path
        assert page.problems == []
    student = open_page(world.url, "/app/", world.student, viewport=PHONE)
    assigned_card(student).wait_for()
    assert student.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
