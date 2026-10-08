"""Authentication at the API boundary: principals, the gateway and static-token authenticators, fail-closed
configuration, and the learner-facing response schemas."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.auth import (
    AUTHOR,
    GatewayAuthenticator,
    Principal,
    StaticTokenAuthenticator,
    authenticator_from_settings,
    gateway_headers,
)
from app.config.routing import ConfigError
from app.config.settings import Settings
from app.schemas.artifact import Artifact, ArtifactType, ArtifactView
from app.schemas.learner import (
    AnswerEvaluation,
    AssessmentRecord,
    LearnerProfile,
    LearnerProfileInput,
    LearnerProfileView,
    MistakeRecord,
)
from app.schemas.task import ArtifactSummary, Task, TaskResult, TaskView, WaitRequest
from app.schemas.workflow import WorkflowState
from tests.auth import TestAuth

SECRET = "s" * 40
NOW = 1_800_000_000


def gateway(**kw) -> GatewayAuthenticator:
    return GatewayAuthenticator(SECRET, clock=lambda: NOW, **kw)


@pytest.fixture
def gateway_app(container):
    container.learner_service.upsert("learner-a", LearnerProfileInput(display_name="A"))
    with TestClient(create_app(container, authenticator=GatewayAuthenticator(SECRET))) as c:
        yield c


def signed(path: str, *, method: str = "GET", learners=("learner-a",), roles=(), secret: str = SECRET, **kw):
    return gateway_headers(secret, method=method, path=path, user_id="user-1", learner_ids=list(learners),
                           roles=list(roles), **kw)


# --- principals ------------------------------------------------------------------------------------------------


def test_a_principal_reaches_only_its_learner_scope() -> None:
    p = Principal(user_id="u", learner_ids=frozenset({"a", "b"}), roles=frozenset({AUTHOR}))
    assert p.may_access("a") and p.may_access("b") and not p.may_access("c") and not p.may_access(None)
    assert p.has_role(AUTHOR)
    with pytest.raises(ValueError):
        Principal(user_id="")
    with pytest.raises(ValueError):
        Principal(user_id="u", roles=frozenset({"superuser"}))


# --- the gateway authenticator (production) --------------------------------------------------------------------


def test_gateway_headers_round_trip(gateway_app: TestClient) -> None:
    r = gateway_app.get("/learners/learner-a", headers=signed("/learners/learner-a"))
    assert r.status_code == 200 and r.json()["display_name"] == "A"
    other = gateway_app.get("/learners/learner-b", headers=signed("/learners/learner-b"))
    assert other.status_code == 404  # authenticated, but learner-b is outside the scope the gateway asserted


@pytest.mark.parametrize("tamper", ["learners", "roles", "subject", "path", "method", "secret", "signature",
                                    "expired", "future", "timestamp", "missing"])
def test_gateway_rejects_anything_it_did_not_sign(gateway_app: TestClient, tamper: str) -> None:
    path = "/learners/learner-a"
    headers = signed(path)
    if tamper == "learners":
        headers["x-auth-learners"] = "learner-a,learner-b"
    elif tamper == "roles":
        headers["x-auth-roles"] = AUTHOR
    elif tamper == "subject":
        headers["x-auth-subject"] = "user-2"
    elif tamper == "path":
        headers = signed("/learners/learner-a/progress")
    elif tamper == "method":
        headers = signed(path, method="PUT")
    elif tamper == "secret":
        headers = signed(path, secret="t" * 40)
    elif tamper == "signature":
        headers["x-auth-signature"] = "0" * 64
    elif tamper == "expired":
        headers = signed(path, timestamp=int(time.time()) - 3600)
    elif tamper == "future":
        headers = signed(path, timestamp=int(time.time()) + 3600)
    elif tamper == "timestamp":
        headers["x-auth-timestamp"] = "yesterday"
    elif tamper == "missing":
        headers.pop("x-auth-signature")
    r = gateway_app.get(path, headers=headers)
    assert r.status_code == 401 and r.json() == {"error": "Unauthenticated", "detail": "authentication required"}


def test_gateway_maps_roles_and_tenant() -> None:
    from starlette.requests import Request

    headers = gateway_headers(SECRET, method="POST", path="/assessment-items", user_id="teacher",
                              learner_ids=["a", "b"], roles=[AUTHOR], tenant_id="school-1", timestamp=NOW)
    scope = {"type": "http", "method": "POST", "path": "/assessment-items", "query_string": b"",
             "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    p = gateway().authenticate(Request(scope))
    assert p == Principal(user_id="teacher", learner_ids=frozenset({"a", "b"}), roles=frozenset({AUTHOR}),
                          tenant_id="school-1")


def test_a_weak_gateway_secret_is_refused() -> None:
    with pytest.raises(ConfigError):
        GatewayAuthenticator("short")


# --- static tokens (development and tests) ---------------------------------------------------------------------


def test_static_tokens(container, tmp_path) -> None:
    container.learner_service.upsert("learner-a", LearnerProfileInput(display_name="A"))
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"dev-token-for-learner-a-0001": {"user_id": "dev", "learner_ids": ["learner-a"]}}))
    auth = StaticTokenAuthenticator.from_file(tokens)
    with TestClient(create_app(container, authenticator=auth)) as c:
        ok = c.get("/learners/learner-a", headers={"Authorization": "Bearer dev-token-for-learner-a-0001"})
        assert ok.status_code == 200
        for bad in ("Bearer dev-token-for-learner-a-0002", "Bearer dev-token-for-learner-a-000", "dev-token",
                    "Basic ZGV2OmRldg=="):
            assert c.get("/learners/learner-a", headers={"Authorization": bad}).status_code == 401
    with pytest.raises(ConfigError):
        StaticTokenAuthenticator({"short": Principal(user_id="u")})
    tokens.write_text("{not json")
    with pytest.raises(ConfigError):
        StaticTokenAuthenticator.from_file(tokens)


# --- fail closed -----------------------------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("TA_AUTH_MODE", "TA_AUTH_GATEWAY_SECRET", "TA_AUTH_STATIC_TOKENS_FILE"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_the_default_configuration_refuses_to_start_without_a_gateway_secret(clean_env) -> None:
    assert Settings().auth_mode == "gateway"  # production is the default, never anonymous
    with pytest.raises(ConfigError, match="TA_AUTH_GATEWAY_SECRET"):
        create_app()
    with pytest.raises(ConfigError, match="TA_AUTH_STATIC_TOKENS_FILE"):
        authenticator_from_settings(Settings(auth_mode="static"))
    with pytest.raises(ValueError):  # there is no anonymous mode to select
        Settings(auth_mode="anonymous")


def test_the_configured_authenticator_is_used(clean_env, tmp_path) -> None:
    clean_env.setenv("TA_AUTH_GATEWAY_SECRET", SECRET)
    assert isinstance(authenticator_from_settings(Settings()), GatewayAuthenticator)
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"x" * 20: {"user_id": "dev", "learner_ids": ["a"]}}))
    clean_env.setenv("TA_AUTH_MODE", "static")
    clean_env.setenv("TA_AUTH_STATIC_TOKENS_FILE", str(tokens))
    assert isinstance(authenticator_from_settings(Settings()), StaticTokenAuthenticator)
    assert "x" * 20 not in repr(Settings()) and SECRET not in repr(Settings())


def test_an_app_without_an_authenticator_rejects_everything(container) -> None:
    app = create_app(container, authenticator=TestAuth().authenticator)
    app.state.authenticator = None
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/providers").status_code == 401


# --- learner-facing schemas ------------------------------------------------------------------------------------

FORBIDDEN_FIELDS = {"expected_answer", "accepted_answers", "acceptable_answers", "answer_key", "correct_answer",
                    "reference_answer", "known_errors", "expected", "answer_hash", "workflow", "provider_requests",
                    "node_states", "uri"}
AUTHOR_FACING = {("post", "/assessment-items")}  # the item's author registers (and gets back) the answer key


def test_no_learner_facing_response_schema_has_an_answer_key_or_internal_field() -> None:
    """A schema walker over every response model the API serves (free-form dicts are checked at runtime by the
    security e2e tests)."""
    spec = create_app(authenticator=TestAuth().authenticator).openapi()
    components = spec["components"]["schemas"]

    def fields(schema, seen: set[str]) -> set[str]:
        if isinstance(schema, list):
            return {f for s in schema for f in fields(s, seen)}
        if not isinstance(schema, dict):
            return set()
        if "$ref" in schema:
            name = schema["$ref"].rsplit("/", 1)[-1]
            return set() if name in seen else fields(components[name], seen | {name})
        found = set(schema.get("properties", {}))
        return found | {f for v in schema.values() for f in fields(v, seen)}

    checked = 0
    for path, operations in spec["paths"].items():
        for method, op in operations.items():
            if (method, path) in AUTHOR_FACING:
                continue
            ok = [r for status, r in op["responses"].items() if status.startswith("2")]
            assert not fields(ok, set()) & FORBIDDEN_FIELDS, (method, path, fields(ok, set()) & FORBIDDEN_FIELDS)
            checked += 1
    assert checked >= 40


def test_task_view_drops_internal_state() -> None:
    task = Task(task_id="t", user_id="u", learner_id="l", request="r",
                waiting=WaitRequest(node_id="answers_1", kind="diagnostic_answers", prompt={"questions": []}),
                workflow=WorkflowState(workflow_id="w"), metadata={"secret": "x"},
                result=TaskResult(title="T", mastery_changes=[], artifacts=[ArtifactSummary(
                    artifact_id="a", type=ArtifactType.LESSON, name="lesson", version=1,
                    uri="file:///srv/data/objects/x", parent_ids=[])]))
    view = TaskView.of(task).model_dump(mode="json")
    assert set(view) == {"task_id", "learner_id", "request", "status", "current_step", "waiting", "errors",
                         "artifact_ids", "cost", "result", "created_at", "updated_at"}
    assert view["waiting"] == {"kind": "diagnostic_answers", "prompt": {"questions": []}}
    assert "uri" not in json.dumps(view) and "secret" not in json.dumps(view)
    assert set(view["cost"]) == {"estimated_cost_usd", "actual_cost_usd"}


def test_artifact_view_drops_storage_locations() -> None:
    artifact = Artifact(artifact_id="a", task_id="t", type=ArtifactType.IMAGE_ASSET, name="img", uri="file:///srv/x",
                        media_type="image/png", content_hash="h", size_bytes=1, version=1, provider="mock",
                        metadata={"object": {"uri": "file:///srv/y", "checksum": "c"}, "items": [{"uri": "z"}]})
    view = ArtifactView.of(artifact).model_dump(mode="json")
    assert "uri" not in json.dumps(view) and view["metadata"] == {"object": {"checksum": "c"}, "items": [{}]}


def test_learner_profile_view_drops_the_answers_it_was_graded_against() -> None:
    graded = AnswerEvaluation(question_id="q", concept_id="c", answer="hablaba", expected="hablé", correct=False,
                              difficulty=0.4, feedback="Expected: hablé")
    profile = LearnerProfile(learner_id="l", assessments=[AssessmentRecord(task_id="t", subject="spanish",
                                                                            evaluations=[graded])],
                             mistakes=[MistakeRecord(task_id="t", concept_id="c", question_id="q", answer="hablaba",
                                                     expected="hablé")])
    view = LearnerProfileView.of(profile).model_dump(mode="json")
    assert "hablé" not in json.dumps(view, ensure_ascii=False)
    assert view["mistakes"][0]["answer"] == "hablaba" and view["assessments"][0]["evaluations"][0]["correct"] is False
    assert profile.mistakes[0].expected == "hablé"  # the stored profile keeps it
