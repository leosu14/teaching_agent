"""MVP item 6: the student web app's static files and the pieces of the lesson reading that need no lesson run."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.routes import web
from app.schemas.artifact import ArtifactType
from app.schemas.lesson import SectionVisual
from app.services.lesson_reading import LessonReadingService, _Filter, _reference, answer_strings, sentences
from app.storage.repositories import NotFound
from tests.auth import TestAuth, client_for

WEB_ROUTES = {"/app", "/app/", "/app/{name}"}


def api_routes(routes) -> list[APIRoute]:
    """Every APIRoute, also inside included routers (FastAPI wraps them)."""
    out = []
    for r in routes:
        if isinstance(r, APIRoute):
            out.append(r)
        elif hasattr(r, "original_router"):
            out += api_routes(r.original_router.routes)
    return out


@pytest.fixture
def anonymous(container):
    with TestClient(create_app(container, authenticator=TestAuth().authenticator)) as c:
        yield c


def test_the_app_files_are_served_without_credentials_and_with_a_strict_policy(anonymous) -> None:
    root = anonymous.get("/app", follow_redirects=False)
    assert root.status_code == 308 and root.headers["location"] == "/app/"
    for path, media in [("/app/", "text/html"), ("/app/app.js", "text/javascript"), ("/app/app.css", "text/css"),
                        ("/app/teacher.html", "text/html"), ("/app/teacher.js", "text/javascript"),
                        ("/app/privacy.html", "text/html")]:
        r = anonymous.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith(media), path
        assert r.headers["content-security-policy"] == web.CSP
        assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"
    assert "script-src 'self';" in web.CSP and "unsafe" not in web.CSP and "frame-ancestors 'none'" in web.CSP


@pytest.mark.parametrize("path", ["/app/index.htm", "/app/..%2Fapp.py", "/app/%2e%2e/routes/web.py", "/app/app.js.map",
                                  "/app/web.py", "/app/x/app.js"])
def test_nothing_but_the_listed_files_is_served(anonymous, path) -> None:
    r = anonymous.get(path)
    assert r.status_code == 404 and "import" not in r.text


def test_only_health_and_the_app_files_are_public(container) -> None:
    """The security matrix (tests/e2e/test_security_api.py) covers every route in the OpenAPI schema; the app's file
    routes are the only ones outside it, and every protected route still answers 401 without credentials."""
    app = create_app(container, authenticator=TestAuth().authenticator)
    routes = api_routes(app.routes)
    assert len(routes) > 50  # the walk reaches into the included routers
    hidden = {r.path for r in routes if not r.include_in_schema}
    assert hidden == WEB_ROUTES
    with TestClient(app) as c:
        assert c.get("/me").status_code == 401 and c.get("/courses").status_code == 401
        assert c.get("/lessons/anything/reading").status_code == 401


@pytest.mark.parametrize("page,script", [("index.html", "app.js"), ("teacher.html", "teacher.js")])
def test_the_page_has_no_inline_script_markup_injection_or_third_party_origin(page, script) -> None:
    """The student app and the teacher dashboard (MVP item 7) follow the same rules."""
    html, js, css = [(web.WEB_DIR / n).read_text(encoding="utf-8") for n in (page, script, "app.css")]
    assert re.findall(r"<script\b[^>]*>", html) == [f'<script src="{script}" defer>']  # no inline script
    assert " style=" not in html and not re.search(r"\son[a-z]+\s*=", html)  # no inline style or handler
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in js, sink  # text is only ever set with textContent
    assert 'setAttribute("style"' not in js and "style:" not in js  # the CSP allows no inline style attribute
    for text in (html, js, css):
        assert not re.search(r"https?://", text)  # no CDN, font or analytics origin


def test_the_teacher_dashboard_sends_no_change_of_a_learners_data() -> None:
    """The dashboard reads learners' data and changes only the teacher's own classes: its only writes are creating a
    class and rotating a join code, and it never asks for the next action (a read that stores progress)."""
    js = (web.WEB_DIR / "teacher.js").read_text(encoding="utf-8")
    assert re.findall(r'\bpost\(`?"?([^`",]+)', js) == ["/classes", "/classes/${enc(classId)}/join-code"]
    assert set(re.findall(r'api\("([A-Z]+)"', js)) == {"GET", "POST"}
    assert js.count('api("POST"') == 1  # only inside the `post` helper: every other request is a GET
    assert "next-action" not in js and "/responses" not in js and "/attempts" not in js


def test_identity_is_the_callers_own(container) -> None:
    with client_for(container, "l2", "l1", user_id="guardian", roles=("author",)) as c:
        assert c.get("/me").json() == {"user_id": "guardian", "learner_ids": ["l1", "l2"], "roles": ["author"]}


# --- the reading's filters -----------------------------------------------------------------------------------------


def test_sentences_split_on_sentence_ends_only() -> None:
    assert sentences("Ayer hablé. ¿Y tú? ¡Sí! Vale") == ["Ayer hablé.", "¿Y tú?", "¡Sí!", "Vale"]
    assert sentences("") == [] and sentences(None) == []
    assert sentences("Dr Smith said 3.5 things.") == ["Dr Smith said 3.5 things."]


def test_answer_strings_finds_every_answer_key_field_at_any_depth() -> None:
    record = {"workflow": {"node_states": {"diagnose": {"output": {"items": [
        {"question": {"prompt": "p"}, "expected_answer": "hablé", "accepted_answers": ["hable"]}]}}}},
        "assess": [{"expected_answer": "fui", "acceptable_answers": ["fue"], "answer": "learner text"}]}
    assert sorted(answer_strings(record)) == ["fue", "fui", "hable", "hablé"]  # never the learner's own answer


def test_the_filter_drops_whole_sentences_and_counts_them() -> None:
    f = _Filter(["cenamos", "", "  "])
    assert f.sentences("We ate out last night. Anoche cenamos fuera. It was fun.") == [
        "We ate out last night.", "It was fun."]
    assert f.text("Cenamos!") is None and f.text("cenar") == "cenar"  # whole words, any case and punctuation
    assert f.items(["cenamos juntos", "comimos"]) == ["comimos"]
    assert f.withheld == 3


def test_a_reference_link_is_http_only() -> None:
    f = _Filter(["secreto"])
    assert _reference("Grammar", "Pub", "https://example.org/a", f).url == "https://example.org/a"
    assert _reference("Grammar", None, "kb://course/concept/reference", f).url is None
    assert _reference("Grammar", None, "javascript:alert(1)", f).url is None
    assert _reference("El secreto", None, "https://example.org", f) is None


def test_only_servable_images_of_the_lesson_task_are_listed() -> None:
    def artifact(aid, task_id="task_1", type_=ArtifactType.IMAGE_ASSET, planned=True):
        return SimpleNamespace(artifact_id=aid, task_id=task_id, type=type_,
                               servable=planned)

    stored = {a.artifact_id: a for a in [
        artifact("img_ok"), artifact("img_other_task", task_id="task_2"), artifact("img_unplanned", planned=False),
        artifact("svg", type_=ArtifactType.SLIDE)]}

    def get(aid):
        if aid not in stored:
            raise NotFound(aid)
        return stored[aid]

    service = LessonReadingService(artifacts=SimpleNamespace(get=get), tasks=None, teaching=None, assessment=None,
                                   content=SimpleNamespace(servable=lambda a: a.servable))

    def visual(aid, description="A street at night"):
        return SectionVisual(visual_id=f"v_{aid}", artifact_id=aid, asset_id="a", visual_type="photo",
                             origin="generated", purpose="illustrate", description=description)

    f = _Filter(["anoche"])
    images = service._images([visual("img_ok"), visual("img_other_task"), visual("img_unplanned"), visual("svg"),
                              visual("missing"), visual("img_ok", description="Anoche en la calle")], "task_1", f)
    assert [(i.artifact_id, i.description) for i in images] == [("img_ok", "A street at night"), ("img_ok", None)]
    assert f.withheld == 4  # three refused images and one description stating an answer
