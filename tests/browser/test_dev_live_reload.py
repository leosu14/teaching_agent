"""Development live reload in a real browser (TA_DEV_LIVE_RELOAD=true, app/api/routes/dev_reload.py).

An open page of the student app or the teacher dashboard reloads by itself once an edit to its HTML, CSS or JS is
being served, keeps its `#/` route and its sign-in, reloads once per edit (no loop), and rides out a server restart
without reloading. The files are a copy of app/api/web/, so the test edits nothing in the repository."""

from __future__ import annotations

import shutil
import socket
import threading
import time

import pytest

from app.api.app import create_app
from app.api.auth import TEACHER
from app.api.routes import web
from app.config.settings import Settings
from app.services.container import build_container
from tests.auth import TestAuth

playwright_api = pytest.importorskip("playwright.sync_api")

TIMEOUT = 15000
RELOADED = "() => window.__beforeEdit === undefined"


class Server:
    def __init__(self, app, port: int) -> None:
        import uvicorn

        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("the test server did not start")
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def live(tmp_path, mock_llm, monkeypatch):
    web_dir = tmp_path / "web"
    shutil.copytree(web.WEB_DIR, web_dir)
    monkeypatch.setattr(web, "WEB_DIR", web_dir)
    container = build_container(Settings(_env_file=None, data_dir=tmp_path / "data", log_json=False,
                                         dev_live_reload=True), llm_providers={"mock": mock_llm})
    auth = TestAuth()
    student = auth.headers("dev-learner")["Authorization"].removeprefix("Bearer ")
    teacher = auth.headers(user_id="dev-teacher", roles=(TEACHER,))["Authorization"].removeprefix("Bearer ")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    def start() -> Server:
        return Server(create_app(container, authenticator=auth.authenticator), port)

    class Live:
        url = f"http://127.0.0.1:{port}"
        dir = web_dir

    Live.tokens = {"index.html": student, "teacher.html": teacher}
    Live.server = start()
    Live.start = staticmethod(start)
    yield Live
    Live.server.stop()
    container.close()


@pytest.fixture
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - Playwright is installed but its browser is not
            pytest.skip(f"Chromium for Playwright is not installed: {exc}")
        yield b
        b.close()


def _open(browser, live, page_name: str, route: str):
    page = browser.new_page()
    page.add_init_script(f"window.sessionStorage.setItem('ta.token', {live.tokens[page_name]!r});")
    path = "/app/" if page_name == "index.html" else f"/app/{page_name}"
    page.goto(f"{live.url}{path}{route}")
    page.wait_for_selector("#nav:not([hidden])", timeout=TIMEOUT)  # signed in: the token was accepted
    page.evaluate("window.__beforeEdit = true")
    return page


def _edit(live, name: str, text: str) -> None:
    path = live.dir / name
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def _stays(page, seconds: float = 2.5) -> None:
    page.wait_for_timeout(seconds * 1000)
    assert page.evaluate("window.__beforeEdit") is True, "the page reloaded without an edit"


@pytest.mark.parametrize("page_name,route,script", [("index.html", "#/progress", "app.js"),
                                                     ("teacher.html", "#/", "teacher.js")])
def test_each_edit_reloads_the_page_once_on_its_route(browser, live, page_name, route, script) -> None:
    page = _open(browser, live, page_name, route)
    _stays(page)  # nothing changed: no reload
    for name, text, check in [
        ("app.css", "\nbody { outline: 1px solid rgb(1, 2, 3); }\n",
         "getComputedStyle(document.body).outlineColor === 'rgb(1, 2, 3)'"),
        (script, "\ndocument.documentElement.dataset.edited = 'js';\n",
         "document.documentElement.dataset.edited === 'js'"),
        (page_name, "<!-- edited -->\n", "document.documentElement.outerHTML.includes('edited')"),
    ]:
        _edit(live, name, text)
        page.wait_for_function(RELOADED, timeout=TIMEOUT)
        page.wait_for_function(f"() => {check}", timeout=TIMEOUT)  # the reloaded page runs the edited file
        assert page.url.endswith(route)  # the route is kept
        page.wait_for_selector("#nav:not([hidden])", timeout=TIMEOUT)  # and the sign-in
        page.evaluate("window.__beforeEdit = true")
        _stays(page)  # one reload per edit
    page.close()


def test_a_server_restart_does_not_reload_the_page_but_a_later_edit_does(browser, live) -> None:
    page = _open(browser, live, "index.html", "#/history")
    live.server.stop()
    page.wait_for_timeout(2500)  # polls fail while the server is down
    assert page.evaluate("window.__beforeEdit") is True
    live.server = live.start()
    _stays(page)  # the same files: no reload after the restart
    _edit(live, "app.js", "\n// edited after the restart\n")
    page.wait_for_function(RELOADED, timeout=TIMEOUT)
    assert page.url.endswith("#/history")
    page.close()
