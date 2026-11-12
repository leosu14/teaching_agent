"""The identity gateway in front of the real API (MVP item 8): the API authenticates only the gateway's signed
headers, so everything here goes through token verification, header stripping and signing, and the API's own
authorization rules (learner scope, the teacher's rosters, answer isolation) still decide what each caller sees."""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.api.auth import TEACHER
from app.gateway.app import create_gateway
from tests.auth import drain
from tests.conftest import ANSWERS, LEARNER
from tests.gateway_fixtures import GATEWAY_SECRET, IdP, gateway_for, gateway_settings

LID = LEARNER["learner_id"]


@pytest.fixture
def idp(tmp_path):
    provider = IdP()
    provider.jwks_path = tmp_path / "jwks.json"
    provider.write_jwks(provider.jwks_path)
    return provider


@pytest.fixture
def gateway(container, idp):
    with TestClient(gateway_for(container, idp.jwks_path)) as client:
        yield client


def bearer(idp: IdP, sub: str = LID, **claims) -> dict:
    return {"Authorization": f"Bearer {idp.token(sub, **claims)}"}


def test_public_paths_need_no_token(gateway) -> None:
    assert gateway.get("/health").json() == {"status": "ok"}
    assert gateway.get("/app/").status_code == 200
    assert gateway.get("/gateway/health").json()["status"] == "ok"


def test_without_a_verified_token_nothing_reaches_the_api(gateway, idp) -> None:
    for headers in ({}, {"Authorization": "Bearer junk"}, bearer(idp, lifetime=-120),
                    {"Authorization": f"Bearer {IdP().token(LID)}"},  # signed by someone else's key
                    {"x-auth-subject": LID, "x-auth-learners": LID, "x-auth-timestamp": "1",
                     "x-auth-signature": "0" * 64}):
        r = gateway.get(f"/learners/{LID}", headers=headers)
        assert r.status_code == 401 and r.json() == {"error": "Unauthorized", "detail": "sign in to continue"}
        assert r.headers["www-authenticate"] == "Bearer"


def test_a_learner_works_through_the_gateway_and_cannot_widen_their_scope(gateway, container, idp) -> None:
    me = bearer(idp)
    assert gateway.put(f"/learners/{LID}", json=LEARNER["profile"], headers=me).status_code == 200
    assert gateway.get(f"/learners/{LID}", headers=me).json()["learner_id"] == LID
    # Identity headers the client adds are dropped: the scope is the token's, signed by the gateway.
    widened = me | {"x-auth-learners": f"{LID},victim", "x-auth-roles": "teacher"}
    assert gateway.get("/learners/victim", headers=widened).status_code == 404
    created = gateway.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}, headers=me)
    assert created.status_code == 202
    drain(container)
    task = gateway.get(f"/tasks/{created.json()['task_id']}", headers=me).json()
    assert task["status"] == "WAITING"
    assert all("expected_answer" not in q for q in task["waiting"]["prompt"]["questions"])  # answer isolation
    other = bearer(idp, "someone-else")
    assert gateway.get(f"/tasks/{task['task_id']}", headers=other).status_code == 404
    assert gateway.get(f"/learners/{LID}", headers=other).status_code == 404


def test_a_teacher_role_from_the_identity_provider_reaches_the_dashboard(gateway, idp) -> None:
    me, teacher = bearer(idp), bearer(idp, "teacher-1", ta_roles=["teacher"])
    gateway.put(f"/learners/{LID}", json=LEARNER["profile"], headers=me)
    assert gateway.post("/classes", json={"name": "No"}, headers=me).status_code in {403, 404}
    made = gateway.post("/classes", json={"name": "Class A"}, headers=teacher)
    assert made.status_code == 201
    joined = gateway.post(f"/learners/{LID}/classes", json={"join_code": made.json()["join_code"]}, headers=me)
    assert joined.status_code == 201
    dashboard = gateway.get(f"/classes/{made.json()['class_id']}/dashboard", headers=teacher)
    assert dashboard.status_code == 200 and LID in dashboard.text
    # The teacher reads the roster's learner but never changes them (the API's rule, unchanged).
    assert gateway.get(f"/learners/{LID}", headers=teacher).status_code == 200
    assert gateway.put(f"/learners/{LID}", json=LEARNER["profile"], headers=teacher).status_code in {403, 404}
    assert TEACHER == "teacher"


def test_a_login_proxys_token_header_is_accepted(gateway, idp) -> None:
    """Behind a login proxy (oauth2-proxy and the like) the browser has a session cookie, and the proxy forwards
    the signed-in user's token in X-Forwarded-Access-Token; the gateway verifies it the same way."""
    headers = {"X-Forwarded-Access-Token": idp.token(LID), "Cookie": "_oauth2_proxy=session"}
    assert gateway.put(f"/learners/{LID}", json=LEARNER["profile"], headers=headers).status_code == 200


def test_what_the_api_receives(idp) -> None:
    """A recording upstream: the token, cookies and client identity headers never reach the API; the signed identity
    covers the method and the path; the query and the body are forwarded unchanged."""
    seen = {}

    async def record(request: Request):
        seen["headers"] = dict(request.headers)
        seen["path"], seen["query"] = request.url.path, request.url.query
        seen["body"] = await request.body()
        return JSONResponse({"ok": True}, headers={"Connection": "close", "X-Upstream": "1"})

    upstream = Starlette(routes=[Route("/{p:path}", record, methods=["GET", "POST"])])
    app = create_gateway(gateway_settings(idp.jwks_path), upstream_transport=httpx.ASGITransport(app=upstream))
    with TestClient(app) as client:
        token = idp.token(LID, ta_roles=["teacher"])
        r = client.post("/tasks/t%201/answers?x=1&y=2", content=b'{"answers": []}',
                        headers={"Authorization": f"Bearer {token}", "Cookie": "s=1", "X-Auth-Roles": "author",
                                 "X-Forwarded-For": "1.2.3.4", "Content-Type": "application/json"})
    assert r.status_code == 200 and r.headers["x-upstream"] == "1"
    h = seen["headers"]
    assert "authorization" not in h and "cookie" not in h and "x-forwarded-for" not in h
    assert token not in str(h)
    assert (h["x-auth-subject"], h["x-auth-learners"], h["x-auth-roles"]) == (LID, LID, "teacher")
    assert seen["path"] == "/tasks/t 1/answers" and seen["query"] == "x=1&y=2" and seen["body"] == b'{"answers": []}'
    from app.api.auth import gateway_headers

    expected = gateway_headers(GATEWAY_SECRET, method="POST", path="/tasks/t 1/answers", user_id=LID,
                               learner_ids=[LID], roles=["teacher"], timestamp=int(h["x-auth-timestamp"]))
    assert h["x-auth-signature"] == expected["x-auth-signature"]


def test_an_unreachable_api_is_a_502_without_details(idp) -> None:
    def refuse(request):
        raise httpx.ConnectError("connection refused to 10.0.0.5:8000")

    app = create_gateway(gateway_settings(idp.jwks_path), upstream_transport=httpx.MockTransport(refuse))
    with TestClient(app) as client:
        r = client.get(f"/learners/{LID}", headers=bearer(idp))
    assert r.status_code == 502 and "10.0.0.5" not in r.text


def test_diagnostic_answers_flow_through_the_gateway(gateway, container, idp) -> None:
    me = bearer(idp)
    gateway.put(f"/learners/{LID}", json=LEARNER["profile"], headers=me)
    created = gateway.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}, headers=me)
    drain(container)
    task = gateway.get(f"/tasks/{created.json()['task_id']}", headers=me).json()
    while task["status"] == "WAITING":
        sheet = task["waiting"]["prompt"]
        key = ANSWERS["rounds"][sheet["round_number"] - 1]
        answers = [{"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]
        assert gateway.post(f"/tasks/{task['task_id']}/answers", json={"answers": answers},
                            headers=me).status_code == 202
        drain(container)
        task = gateway.get(f"/tasks/{task['task_id']}", headers=me).json()
    assert task["status"] == "COMPLETED"
    artifacts = gateway.get(f"/tasks/{task['task_id']}/artifacts", headers=me).json()
    assert artifacts and all("uri" not in a and "content_hash" not in a for a in artifacts)
