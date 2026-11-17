"""Issue #54: the student app's live progress in a real Chromium, over the real API's progress stream.

- While the diagnostic's answers generate the lesson (the response POST runs the lesson inline), the page shows the
  stream's progress lines, then reaches the session.
- A stream cut in the middle of the lesson (once by the server's maximum lifetime, `reconnect`, and once by a dropped
  connection) resumes with `Last-Event-ID`, and no progress line is shown twice.
- With the stream route failing, the page falls back to polling and still gets there.
- A deletion from another tab while the page follows a cycle sends the page to its error state."""

from __future__ import annotations

import asyncio
import json

import httpx

from scripts.run_learning_cycle_demo import ScriptedLearner
from tests.browser.test_student_app import answer_sheet, page_for, site  # noqa: F401 - fixtures
from tests.learning_cycle_fixtures import LEARNER

FAST = {"stream_settle_seconds": 0.2, "stream_poll_seconds": 0.05}


def slow_llm(env, seconds: float = 0.15) -> None:
    """Each model call takes a moment, so the lesson takes long enough to watch it being generated."""
    generate = env.llm.generate

    async def slow(request):
        await asyncio.sleep(seconds)
        return await generate(request)

    env.llm.generate = slow


def open_cycle(env, site, page_for):  # noqa: F811
    url, token = site(env.container, LEARNER)
    page = page_for(url, token)
    [cycle] = env.container.learning_cycle_service.repository.for_learner(LEARNER)
    page.goto(f"{url}/app/#/cycle/{cycle.cycle_id}")
    return page, url, token, cycle


def lines(page) -> list[str]:
    """The shown progress lines' event ids, read in one step (the page may re-render between two reads)."""
    return page.eval_on_selector_all("[data-testid=progress-line]", "els => els.map(e => e.dataset.eventId)")


def test_progress_is_shown_while_the_lesson_is_generated(cycle_env_at, site, page_for) -> None:  # noqa: F811
    env = cycle_env_at("diagnostic", **FAST)
    slow_llm(env)
    page, *_ = open_cycle(env, site, page_for)
    page.get_by_test_id("diagnostic").wait_for()
    streams = []
    page.on("request", lambda r: streams.append(r) if "/stream" in r.url else None)
    answer_sheet(page, ScriptedLearner(prefix="b"))
    page.get_by_test_id("progress-line").first.wait_for()  # while the POST still runs
    assert page.get_by_test_id("chat").count() == 0
    seen = lines(page)
    assert seen and len(seen) == len(set(seen))
    assert "Lesson: " in page.get_by_test_id("progress").inner_text()
    page.get_by_test_id("chat").wait_for()  # the POST's response: the session's first question
    assert streams and all("after_version=" in r.url for r in streams)
    assert all("token" not in r.url.lower() and r.headers.get("authorization") for r in streams)
    assert page.problems == []


def test_a_cut_stream_resumes_without_repeating_progress(cycle_env_at, site, page_for) -> None:  # noqa: F811
    env = cycle_env_at("diagnostic", stream_max_seconds=1.0, **FAST)  # `reconnect` every second
    slow_llm(env)
    page, url, token, cycle = open_cycle(env, site, page_for)
    page.get_by_test_id("diagnostic").wait_for()
    cursors, shown = [], []

    def on_request(r):
        if "/stream" in r.url:
            cursors.append(r.headers.get("last-event-id"))

    page.on("request", on_request)
    cut = {"done": False}

    def drop_once(route):
        """The first stream after the lesson started: answered with half a body and no `end` (a dropped stream)."""
        if cut["done"] or not route.request.headers.get("last-event-id"):
            return route.continue_()
        cut["done"] = True
        headers = {"Authorization": f"Bearer {token}", "Last-Event-ID": route.request.headers["last-event-id"]}
        with httpx.Client(timeout=10) as c, c.stream("GET", route.request.url, headers=headers) as r:
            body = b""
            for chunk in r.iter_raw():
                body += chunk
                if body.count(b"event: progress") >= 2:
                    break
        route.fulfill(status=200, headers={"content-type": "text/event-stream"}, body=body)

    page.route("**/stream*", drop_once)
    answer_sheet(page, ScriptedLearner(prefix="b"))
    page.get_by_test_id("progress-line").first.wait_for()
    for _ in range(200):
        if page.get_by_test_id("chat").count():
            break
        shown = lines(page) or shown
        assert len(shown) == len(set(shown)), shown  # never a line twice
        page.wait_for_timeout(50)
    page.get_by_test_id("chat").wait_for()
    assert cut["done"], "the stream was never cut"
    assert len([c for c in cursors if c]) >= 2  # reconnected with Last-Event-ID after the cut and the lifetime
    assert len(shown) == len(set(shown))
    assert page.problems == []


def test_the_page_falls_back_to_polling_when_the_stream_fails(cycle_env_at, site, page_for) -> None:  # noqa: F811
    env = cycle_env_at("diagnostic", **FAST)
    page, *_ = open_cycle(env, site, page_for)
    page.get_by_test_id("diagnostic").wait_for()
    failed = []

    def fail(route):
        failed.append(route.request.url)
        route.fulfill(status=503, headers={"content-type": "application/json"}, body=json.dumps({"error": "x"}))

    page.route("**/stream*", fail)
    answer_sheet(page, ScriptedLearner(prefix="b"))
    page.get_by_test_id("chat").wait_for()  # the POST's own response still renders the session
    assert failed
    assert page.get_by_test_id("error").count() == 0


def test_a_deletion_in_another_tab_sends_the_page_to_its_error_state(cycle_env_at, site, page_for) -> None:  # noqa: F811
    env = cycle_env_at("session", **FAST)
    cycle = running_undriven(env)  # a read would fold the waiting session back in: kept RUNNING
    url, token = site(env.container, LEARNER)
    page = page_for(url, token)
    page.route("**/resume", lambda route: None)  # the page's resume never arrives: nobody drives the cycle
    page.goto(f"{url}/app/#/cycle/{cycle.cycle_id}")
    page.get_by_test_id("progress-line").first.wait_for()  # following the stream
    deleted = httpx.delete(f"{url}/learners/{LEARNER}", headers={"Authorization": f"Bearer {token}"}, timeout=30)
    assert deleted.status_code == 200, deleted.text
    page.get_by_test_id("error").wait_for()
    assert "could not be found" in page.get_by_test_id("error").inner_text()
    assert page.get_by_test_id("progress").count() == 0  # not a frozen progress view


def running_undriven(env):
    """The session-stage cycle as a RUNNING cycle nobody drives, kept so by every read."""
    service = env.container.learning_cycle_service
    [cycle] = service.repository.for_learner(LEARNER)
    from app.curriculum import cycle as engine
    from app.schemas.common import utcnow

    service.repository.apply(engine.receive(cycle, utcnow()), expected_version=cycle.version)

    async def as_stored(cycle_id):
        record = service.cycle(cycle_id)
        return record, await service._view(record)

    service.read = as_stored
    return cycle


def test_a_stream_that_drops_after_each_status_still_falls_back(cycle_env_at, site, page_for) -> None:  # noqa: F811
    """Each connection opens and sends `status`, then closes without `end`: that is a failure every time (no new
    cursor, no end), so the page falls back after its retries instead of reconnecting forever."""
    env = cycle_env_at("session", **FAST)
    cycle = running_undriven(env)
    url, token = site(env.container, LEARNER)
    page = page_for(url, token)
    page.route("**/resume", lambda route: None)  # nobody drives it
    status = json.dumps(env.container.learning_cycle_service.cycle(cycle.cycle_id).model_dump(mode="json"))
    opened = []

    def status_then_drop(route):
        opened.append(route.request.url)
        route.fulfill(status=200, headers={"content-type": "text/event-stream"},
                      body=f"event: status\ndata: {status}\n\n")

    page.route("**/stream*", status_then_drop)
    page.goto(f"{url}/app/#/cycle/{cycle.cycle_id}")
    page.wait_for_timeout(6000)
    settled = len(opened)
    page.wait_for_timeout(4000)
    # STREAM_RETRIES requests (plus one of a rendering the page replaced), with backoff; then no more streams
    assert 1 <= settled <= 4 and len(opened) == settled, opened
    assert page.get_by_test_id("error").count() == 0
