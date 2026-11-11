"""The teacher dashboard end to end in a real browser (MVP item 7).

A teacher creates a class and gets its join code; the learner joins with it in the student app; the teacher then sees
the roster with the learner's progress and the answer the grader could not decide (never its answer key), opens the
learner's page, and sees the answer settle once a later attempt is graded. The pages also work at phone width, and
the error states (not a teacher, a class that is not theirs, a bad token) say what is wrong. Every click goes through
the pages' own scripts against the real API; nothing is stubbed but the model providers."""

from __future__ import annotations

import json

import pytest

from app.api.app import create_app
from app.api.auth import AUTHOR, TEACHER
from scripts.run_assessment_demo import fixture
from tests.auth import TestAuth
from tests.browser.conftest import serve

TIMEOUT = 30000
PHONE = {"width": 390, "height": 844}


def token_for(auth: TestAuth, *learner_ids: str, **kw) -> str:
    return auth.headers(*learner_ids, **kw)["Authorization"].removeprefix("Bearer ")


@pytest.fixture
def world(teaching_env):
    """The teaching fixture's learner (a generated lesson) behind a served app; tokens for them and for a teacher.
    `world.answer(kind)` submits the fixture's free-text answer of that kind as the learner (over HTTP)."""
    c = teaching_env.container
    learner = c.task_service.get(teaching_env.lesson_task).learner_id
    auth = TestAuth()
    served = serve(create_app(c, authenticator=auth.authenticator))
    entry = next(e for e in fixture()["items"] if e["item"]["assessment_item_id"] == "why_preterite")

    class World:
        url = served.url
        container = c
        learner_id = learner
        name = c.memory.get(learner).display_name
        teacher = token_for(auth, user_id="teacher-web", roles=(TEACHER,))
        student = token_for(auth, learner)
        author = auth.headers(learner, roles=(AUTHOR,))
        item = entry["item"]
        answers = entry["answers"]

        def answer(self, kind: str, attempt_id: str) -> dict:
            from fastapi.testclient import TestClient

            with TestClient(create_app(c, authenticator=auth.authenticator), headers=self.author) as client:
                body = {"item": {**entry["item"], "lesson_id": teaching_env.lesson_task}, "rubric": entry.get("rubric")}
                if body["rubric"] is None:
                    del body["rubric"]
                assert client.post("/assessment-items", json=body).status_code in (200, 201)
                r = client.post(f"/assessment-items/{entry['item']['assessment_item_id']}/attempts",
                                json={"learner_id": learner, "answer": entry["answers"][kind], "attempt_id": attempt_id})
                assert r.status_code == 201, r.text
                return r.json()

    yield World()
    served.stop()


@pytest.fixture
def open_page(browser):
    """`open_page(url, path, token, viewport=None)`: a fresh signed-in context. Console and page errors (CSP
    violations included) are collected in `page.problems`."""
    contexts = []

    def opener(url: str, path: str, token: str | None, viewport: dict | None = None):
        context = browser.new_context(viewport=viewport) if viewport else browser.new_context()
        contexts.append(context)
        page = context.new_page()
        page.set_default_timeout(TIMEOUT)
        page.problems = []  # an expected 404 is logged by the browser as "Failed to load resource": not one
        page.on("console", lambda m: page.problems.append(m.text)
                if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
        page.on("pageerror", lambda e: page.problems.append(str(e)))
        if token is not None:
            page.add_init_script(f"sessionStorage.setItem('ta.token', {json.dumps(token)})")
        page.goto(f"{url}{path}")
        return page

    yield opener
    for c in contexts:
        c.close()


def no_horizontal_scroll(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_a_teacher_creates_a_class_and_reviews_their_learners(world, open_page) -> None:
    assert world.answer("uncertain", "web-unsure")["grade"]["outcome"] == "UNCERTAIN"

    teacher = open_page(world.url, "/app/teacher.html", world.teacher)
    teacher.get_by_text("No classes yet").wait_for()
    teacher.get_by_test_id("class-name").fill("Spanish B1")
    teacher.get_by_test_id("create-class").click()
    code = teacher.get_by_test_id("join-code").inner_text().strip()
    assert len(code) == 16
    teacher.get_by_text("No learners yet").wait_for()

    student = open_page(world.url, "/app/#/progress", world.student)
    student.get_by_test_id("join-code").fill("not-the-code")
    student.get_by_test_id("join-class").click()
    student.get_by_text("That code was not recognised").wait_for()
    student.get_by_test_id("join-code").fill(code)
    student.get_by_test_id("join-class").click()
    student.get_by_test_id("enrollment").wait_for()
    assert student.get_by_test_id("enrollment").inner_text() == "Spanish B1"

    teacher.reload()
    row = teacher.get_by_test_id("learner-row")
    row.wait_for()
    assert world.name in row.inner_text()
    assert teacher.get_by_test_id("review-count").inner_text() == "1 to review"
    assert teacher.get_by_test_id("learner-review").inner_text() == "1"
    assert "join_code" not in teacher.content() and code not in teacher.locator("main").inner_text()  # shown once
    item = teacher.get_by_test_id("uncertain-item")
    assert item.count() == 1
    text = item.inner_text()
    assert world.item["prompt"] in text and world.answers["uncertain"] in text and "to review" in text
    assert world.item["expected_answer"] not in teacher.locator("main").inner_text()  # never the answer key

    teacher.get_by_test_id("learner-link").click()
    teacher.get_by_test_id("learner-summary").wait_for()
    assert teacher.locator("h1").inner_text() == world.name
    assert teacher.get_by_test_id("concepts").locator("tr").count() >= 1  # mastery by concept
    assert teacher.get_by_test_id("uncertain-item").count() == 1
    assert world.item["expected_answer"] not in teacher.locator("main").inner_text()
    teacher.go_back()
    teacher.get_by_test_id("roster").wait_for()

    # a later graded answer settles it: nothing left to review, but the teacher can still look
    assert world.answer("semantic", "web-graded")["grade"]["outcome"] == "CORRECT"
    teacher.reload()
    teacher.get_by_text("Nothing to review. Every answer was graded.").wait_for()
    assert teacher.get_by_test_id("review-count").inner_text() == "nothing to review"
    assert teacher.get_by_test_id("uncertain-item").count() == 0
    teacher.get_by_test_id("show-settled").check()
    teacher.get_by_text("settled by a later attempt").wait_for()
    assert teacher.problems == [] and student.problems == []


def test_the_dashboard_fits_a_phone(world, open_page) -> None:
    world.answer("uncertain", "web-phone")
    made = world.container.classes.create(teacher_id="teacher-web", tenant_id=None, name="Phone class")
    world.container.classes.join(world.learner_id, made.join_code, tenant_id=None)

    page = open_page(world.url, "/app/teacher.html", world.teacher, viewport=PHONE)
    page.get_by_test_id("class-row").wait_for()
    assert no_horizontal_scroll(page)
    page.get_by_role("link", name="Phone class").click()
    page.get_by_test_id("learner-row").wait_for()
    page.get_by_test_id("uncertain-item").wait_for()
    assert no_horizontal_scroll(page)  # the roster table scrolls inside its card, the page does not
    page.get_by_test_id("learner-link").click()
    page.get_by_test_id("learner-summary").wait_for()
    assert no_horizontal_scroll(page)
    assert page.problems == []


def test_error_states_say_what_is_wrong(world, open_page) -> None:
    not_teacher = open_page(world.url, "/app/teacher.html", world.student)
    not_teacher.get_by_test_id("not-teacher").wait_for()
    assert not_teacher.get_by_role("link", name="Go to the student app").count() == 1

    teacher = open_page(world.url, "/app/teacher.html#/class/cls_not_mine", world.teacher)
    teacher.get_by_test_id("error").wait_for()
    assert "not one of your classes" in teacher.get_by_test_id("error").inner_text()
    other = world.container.classes.create(teacher_id="someone-else", tenant_id=None, name="Theirs")
    teacher.goto(f"{world.url}/app/teacher.html#/class/{other.class_id}")
    teacher.reload()
    teacher.get_by_test_id("error").wait_for()  # another teacher's class: the same answer as none at all
    assert "Theirs" not in teacher.locator("main").inner_text()

    signed_out = open_page(world.url, "/app/teacher.html", "not-a-valid-token-at-all")
    signed_out.get_by_test_id("token").wait_for()

    # the student app sends a teacher-only account to the dashboard
    student_app = open_page(world.url, "/app/", world.teacher)
    student_app.get_by_test_id("teacher-link").click()
    student_app.get_by_text("Your classes").wait_for()
    for page in (not_teacher, teacher, signed_out, student_app):
        assert page.problems == []
