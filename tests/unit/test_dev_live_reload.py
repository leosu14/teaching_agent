"""Development live reload (TA_DEV_LIVE_RELOAD, app/api/routes/dev_reload.py): off by default, refused in production,
and when on, edits to the apps' HTML, CSS and JS are served at once with a new version for the page to reload on."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.routes import dev_reload, web
from app.config.settings import Settings
from app.services.container import build_container
from tests.auth import TestAuth

PAGES = ("/app/", "/app/teacher.html", "/app/privacy.html")


def test_live_reload_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TA_DEV_LIVE_RELOAD", raising=False)
    assert Settings(_env_file=None).dev_live_reload is False


def test_the_default_app_has_no_reload_script_or_endpoint(container) -> None:
    with TestClient(create_app(container, authenticator=TestAuth().authenticator)) as c:
        for page in PAGES:
            r = c.get(page)
            assert r.status_code == 200 and dev_reload.SCRIPT_NAME.encode() not in r.content
            name = page.removeprefix("/app/") or "index.html"
            assert b"dev-reload" not in r.content and r.content == web._CONTENT[name]
        assert c.get(f"/app/{dev_reload.SCRIPT_NAME}").status_code == 404
        assert c.get(dev_reload.VERSION_PATH).status_code == 404
    for name in web.FILES:
        assert b"dev-reload" not in (web.WEB_DIR / name).read_bytes(), name


def test_production_refuses_live_reload() -> None:
    problems = Settings(_env_file=None, deployment="production", dev_live_reload=True).deployment_problems()
    assert any(p.startswith("TA_DEV_LIVE_RELOAD must be false") for p in problems)
    off = Settings(_env_file=None, deployment="production").deployment_problems()
    assert not any("TA_DEV_LIVE_RELOAD" in p for p in off)


@pytest.fixture
def live(tmp_path: Path):
    """The development routes over a copy of app/api/web/, so the test can edit the files."""
    web_dir = tmp_path / "web"
    shutil.copytree(web.WEB_DIR, web_dir)
    app = FastAPI()
    app.include_router(dev_reload.router(web_dir))
    app.include_router(web.router)
    with TestClient(app) as c:
        yield c, web_dir


def _served_version(html: str) -> str:
    marker = f'<script src="{dev_reload.SCRIPT_NAME}" data-version="'
    assert html.count(marker) == 1
    return html.split(marker, 1)[1].split('"', 1)[0]


def test_live_pages_carry_the_script_and_keep_every_security_header(live) -> None:
    c, _ = live
    current = c.get(dev_reload.VERSION_PATH).json()["version"]
    for page in PAGES:
        r = c.get(page)
        assert r.status_code == 200 and _served_version(r.text) == current
        assert r.text.index(dev_reload.SCRIPT_NAME) < r.text.lower().index("</head>")
    for path in (*PAGES, "/app/app.js", f"/app/{dev_reload.SCRIPT_NAME}", dev_reload.VERSION_PATH):
        h = c.get(path).headers
        assert h["content-security-policy"] == web.CSP, path
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY", path
        assert h["referrer-policy"] == "no-referrer" and h["cache-control"] == "no-store", path
    script = c.get(f"/app/{dev_reload.SCRIPT_NAME}")
    assert script.headers["content-type"].startswith("text/javascript")
    assert "innerHTML" not in script.text and "eval" not in script.text and "http" not in script.text


@pytest.mark.parametrize("name", ["index.html", "teacher.html", "app.css", "app.js", "teacher.js"])
def test_an_edit_is_served_at_once_with_a_new_version(live, name) -> None:
    c, web_dir = live
    before = c.get(dev_reload.VERSION_PATH).json()["version"]
    assert c.get(dev_reload.VERSION_PATH).json()["version"] == before  # stable while nothing changes
    path = web_dir / name
    edit = "<!-- edited -->" if name.endswith(".html") else "/* edited */"
    path.write_text(path.read_text(encoding="utf-8") + edit + "\n", encoding="utf-8")
    after = c.get(dev_reload.VERSION_PATH).json()["version"]
    assert after != before
    served = c.get("/app/" if name == "index.html" else f"/app/{name}").text
    assert edit in served
    if name.endswith(".html"):
        assert _served_version(served) == after  # the reloaded page carries the new version: no reload loop


def test_only_the_listed_files_are_served_in_development_too(live) -> None:
    c, web_dir = live
    (web_dir / "secret.txt").write_text("x", encoding="utf-8")
    for path in ("/app/secret.txt", "/app/..%2Fapp.py", "/app/x/app.js", "/app/dev-reload/other"):
        assert c.get(path).status_code == 404, path


def test_create_app_mounts_the_routes_only_when_enabled(tmp_path: Path, mock_llm) -> None:
    container = build_container(Settings(_env_file=None, data_dir=tmp_path / "data", log_json=False,
                                         dev_live_reload=True), llm_providers={"mock": mock_llm})
    try:
        with TestClient(create_app(container, authenticator=TestAuth().authenticator)) as c:
            assert dev_reload.SCRIPT_NAME in c.get("/app/").text
            assert c.get(dev_reload.VERSION_PATH).status_code == 200
            assert c.get("/app", follow_redirects=False).status_code == 308
            assert c.get("/me").status_code == 401  # authentication is unchanged
    finally:
        container.close()


def test_version_polls_are_not_access_logged() -> None:
    f = dev_reload._QuietPolls()
    poll = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                             ("127.0.0.1:1", "GET", dev_reload.VERSION_PATH, "1.1", 200), None)
    page = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                             ("127.0.0.1:1", "GET", "/app/", "1.1", 200), None)
    assert not f.filter(poll) and f.filter(page)
