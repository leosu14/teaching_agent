"""The web apps behind the identity gateway, as deployed (MVP item 8), in a real Chromium.

The API runs in gateway mode (it trusts only the gateway's HMAC-signed identity) and the gateway in front of it
verifies the identity provider's tokens. The browser plays a signed-in user behind a login proxy: every request it
makes carries the proxy's `X-Forwarded-Access-Token` (here added by the browser context, as the proxy would), and no
development token is pasted or stored. The student app must know who it is from `/me`, onboard the learner and save
their profile; the teacher dashboard must create a class; without a sign-in the apps show the sign-in screen."""

from __future__ import annotations

import pytest

from app.api.app import create_app
from app.api.auth import GatewayAuthenticator
from app.gateway.app import create_gateway
from tests.browser.conftest import serve
from tests.gateway_fixtures import GATEWAY_SECRET, IdP, gateway_settings

TIMEOUT = 60000


@pytest.fixture
def deployed(container, tmp_path):
    idp = IdP()
    jwks = tmp_path / "jwks.json"
    idp.write_jwks(jwks)
    api = serve(create_app(container, authenticator=GatewayAuthenticator(GATEWAY_SECRET)))
    gateway = serve(create_gateway(gateway_settings(jwks, upstream_url=api.url)))
    yield gateway.url, idp
    gateway.stop()
    api.stop()


@pytest.fixture
def open_as(browser):
    contexts = []

    def open_page(url: str, path: str, token: str | None):
        context = browser.new_context(extra_http_headers={"X-Forwarded-Access-Token": token} if token else {})
        contexts.append(context)
        page = context.new_page()
        page.set_default_timeout(TIMEOUT)
        page.problems = []
        page.on("console", lambda m: page.problems.append(m.text)
                if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
        page.on("pageerror", lambda e: page.problems.append(str(e)))
        page.goto(f"{url}{path}")
        return page

    yield open_page
    for c in contexts:
        c.close()


def test_a_signed_in_learner_onboards_through_the_gateway(deployed, open_as, container) -> None:
    url, idp = deployed
    page = open_as(url, "/app/", idp.token("gw-learner"))
    page.get_by_test_id("display-name").fill("Bea")
    page.get_by_test_id("save-profile").click()
    page.get_by_test_id("level").wait_for()
    assert container.learner_service.get("gw-learner").display_name == "Bea"
    assert page.evaluate("sessionStorage.getItem('ta.token')") is None  # the session is the proxy's, not a token
    assert page.get_by_test_id("token").count() == 0
    assert page.problems == []


def test_without_a_sign_in_the_apps_ask_for_one(deployed, open_as) -> None:
    url, idp = deployed
    page = open_as(url, "/app/", None)
    page.get_by_test_id("token").wait_for()
    assert "sign" in page.locator("h1").inner_text().lower()
    expired = open_as(url, "/app/", idp.token("gw-learner", lifetime=-300))
    expired.get_by_test_id("token").wait_for()


def test_a_teacher_creates_a_class_through_the_gateway(deployed, open_as, container) -> None:
    url, idp = deployed
    teacher = open_as(url, "/app/teacher.html", idp.token("gw-teacher", ta_roles=["teacher"]))
    teacher.get_by_test_id("class-name").fill("Spanish A2")
    teacher.get_by_test_id("create-class").click()
    assert teacher.get_by_test_id("join-code").inner_text().strip()
    assert [c.name for c in container.classes.for_teacher(teacher_id="gw-teacher", tenant_id=None)] == ["Spanish A2"]
    assert teacher.problems == []
