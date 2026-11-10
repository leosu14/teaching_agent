"""The student web app end to end in a real browser (MVP item 6).

A learner signs in, creates a profile and a goal, plans the path, starts the next action, answers the diagnostic,
practises in the teaching session, reads the lesson, finishes the cycle and sees their progress and history. Every
click goes through the page's own script against the real API; nothing is stubbed but the model providers."""

from __future__ import annotations

import json
import re

import pytest

from app.api.app import create_app
from app.schemas.learner import LearnerProfileInput
from scripts.run_learning_cycle_demo import ScriptedLearner
from tests.auth import TestAuth
from tests.browser.conftest import serve
from tests.learning_cycle_fixtures import LEARNER

TIMEOUT = int(__import__("os").environ.get("TA_BROWSER_TIMEOUT_MS", "60000"))  # a lesson is generated inside the cycle's start request (mock providers: a few seconds)


def token_for(auth: TestAuth, *learner_ids: str, **kw) -> str:
    return auth.headers(*learner_ids, **kw)["Authorization"].removeprefix("Bearer ")


@pytest.fixture
def site():
    """`site(container, *learner_ids)` -> (base url, token); servers stop at teardown."""
    served = []

    def start(container, *learner_ids: str, **kw):
        auth = TestAuth()
        token = token_for(auth, *learner_ids, **kw)
        s = serve(create_app(container, authenticator=auth.authenticator))
        served.append(s)
        return s.url, token

    yield start
    for s in served:
        s.stop()


@pytest.fixture
def page_for(browser):
    """`page_for(url, token)`: a fresh browser context signed in with the token (None: signed out). Console errors
    and page errors (CSP violations included) are collected in `page.problems`."""
    contexts = []

    def open_page(url: str, token: str | None):
        context = browser.new_context()
        contexts.append(context)
        page = context.new_page()
        page.set_default_timeout(TIMEOUT)
        page.problems = []  # an expected 404 or 409 is logged by the browser as "Failed to load resource": not one
        page.on("console", lambda m: page.problems.append(m.text)
                if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
        page.on("pageerror", lambda e: page.problems.append(str(e)))
        if token is not None:
            page.add_init_script(f"sessionStorage.setItem('ta.token', {json.dumps(token)})")
        page.goto(f"{url}/app/")
        return page

    yield open_page
    for c in contexts:
        c.close()


def answer_sheet(page, learner: ScriptedLearner) -> None:
    for q in page.locator(".question").all():
        prompt = re.sub(r"^\d+\.\s*", "", q.locator("p").inner_text())
        q.locator("input[type=text]").fill(learner.answer(prompt))
    page.get_by_test_id("submit-sheet").click()


def test_a_new_learner_onboards_and_sets_a_goal(container, site, page_for) -> None:
    url, token = site(container, "web-new")
    page = page_for(url, token)
    page.get_by_test_id("display-name").fill("Ana")
    page.get_by_test_id("save-profile").click()
    page.get_by_test_id("level").select_option("B1")
    assert page.get_by_test_id("goal-title").input_value() == "Reach B1 Spanish"  # suggested from the choices
    page.get_by_test_id("save-goal").click()
    try:
        page.get_by_test_id("next-action").wait_for()
    except Exception:
        print("MAIN:", page.locator("main").inner_html()[:3000], page.problems)
        raise
    assert page.locator("h1").inner_text() == "Hello, Ana"
    assert page.get_by_test_id("start").is_visible()  # the path was planned with the goal: a LEARN action is next
    assert "Reach B1 Spanish" in page.get_by_test_id("goal").inner_text()
    assert container.learner_service.get("web-new").display_name == "Ana"
    [goal] = container.curriculum_service.goals("web-new")
    assert goal.target_level == "B1" and container.curriculum_service.curriculum(goal.goal_id) is not None
    assert page.problems == []


def test_a_learning_cycle_from_plan_to_progress(cycle_env_at, site, page_for) -> None:
    env = cycle_env_at("base")  # the learner and an active goal without a curriculum
    url, token = site(env.container, LEARNER)
    page = page_for(url, token)

    page.get_by_test_id("plan").click()  # no path yet: the next action is to plan one
    page.get_by_test_id("start").wait_for()
    objective = page.get_by_test_id("objective").inner_text()
    page.get_by_test_id("start").click()

    page.get_by_test_id("diagnostic").wait_for()
    assert page.locator("h1").inner_text() == objective
    learner = ScriptedLearner(prefix="b")
    answer_sheet(page, learner)

    page.get_by_test_id("chat").wait_for()
    lesson_task = env.container.learning_cycle_service.repository.for_learner(LEARNER)[0].steps[0].child_id
    keys = [k for k in _answer_keys(env.container, lesson_task) if len(k) > 3]
    page.get_by_test_id("read-lesson").click()
    page.get_by_test_id("lesson").wait_for()
    text = page.get_by_test_id("lesson").inner_text()
    assert page.get_by_test_id("section").count() >= 1
    assert not [k for k in keys if re.search(rf"\b{re.escape(k)}\b", text, re.I)]  # no answer key in the reading
    page.go_back()
    page.get_by_test_id("chat").wait_for()

    for _ in range(20):  # answer the teacher's questions until the cycle is done
        if page.get_by_test_id("outcome").count():
            break
        question = page.locator(".bubble.teacher").last.inner_text()
        page.get_by_test_id("reply").fill(learner.answer(question))
        before = page.locator(".bubble").count()
        page.get_by_test_id("send").click()
        # (the page's CSP forbids eval, so wait_for_function cannot be used: wait for the next bubble or the outcome)
        page.locator(".bubble").nth(before).or_(page.get_by_test_id("outcome")).first.wait_for()
    outcome = page.get_by_test_id("outcome")
    outcome.wait_for()
    assert "You finished the lesson and the practice." in outcome.inner_text()
    assert "→" in outcome.inner_text()  # the mastery change, before -> after

    page.get_by_test_id("home").click()
    page.get_by_test_id("next-action").wait_for()
    assert page.get_by_test_id("objective").inner_text() != objective  # the curriculum moved on

    page.click("nav a[data-nav=progress]")
    page.get_by_test_id("summary").wait_for()
    assert "Preterite for completed events" in page.get_by_test_id("mastered").inner_text()
    assert page.get_by_test_id("curriculum").count() == 1

    page.click("nav a[data-nav=history]")
    rows = page.get_by_test_id("history").locator("tr")
    rows.first.wait_for()
    assert rows.count() == 1 and "completed" in rows.first.inner_text()
    assert page.problems == []


def test_signing_in_with_a_development_token(container, site, page_for) -> None:
    container.learner_service.upsert("web-signin", LearnerProfileInput(display_name="Bo"))
    url, token = site(container, "web-signin")
    page = page_for(url, None)
    page.get_by_test_id("token").fill("not-the-right-token-at-all")
    page.get_by_test_id("sign-in").click()
    page.get_by_test_id("token").wait_for()  # refused: still signed out
    page.get_by_test_id("token").fill(token)
    page.get_by_test_id("sign-in").click()
    page.get_by_test_id("no-goal").wait_for()
    assert page.locator("h1").inner_text() == "Hello, Bo"
    page.click("#sign-out")
    page.get_by_test_id("token").wait_for()
    assert page.problems == []


def test_an_account_without_a_learner_is_told_so(container, site, page_for) -> None:
    url, token = site(container, user_id="teacher-1", roles=("teacher",))
    page = page_for(url, token)
    page.get_by_text("This account is not linked to a learner").wait_for()
    assert "teacher dashboard" in page.locator("main").inner_text()


def _answer_keys(container, lesson_task: str) -> list[str]:
    from app.agents.visual.briefing import answer_keys
    from app.services.lessons import load_lesson

    return answer_keys(load_lesson(container.artifacts, container.task_service, lesson_task).lesson)
