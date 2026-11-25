"""The live smoke run's own flow (deploy/live_smoke.py) against the real API in process, on the mock providers.

The rehearsal in CI's `deploy` job runs the same script on the compose stack; this one runs it on every test run, in
seconds: `live_smoke.main(["--mock", ...])` with its HTTP client pointed at the application (create_app) instead of
the published gateway, tokens verified as the gateway verifies them (RS256, issuer, audience, expiry; `sub` is the
user and their own learner; `ta_roles` maps to the roles), the worker run as each poll asks, and the script's SQL
answered by the container's own database. Docker, the bucket and ffprobe are the only parts stubbed.

What it proves on the real application paths: the class requires approval and the learner joins it; the one lesson
is held (reading, image and session 409), listed, previewed and its images fetched by its teacher only, approved
through the teacher route, then read; no paid call is made by the class setup or the review (the ledger is
unchanged); the review bundle is sealed to the reviewer's key and opens to the preview, the reading, the release and
the images; the report records each stage's times and the lesson's cost split into measured and estimate, consistent
with the ledger; the run-wide cap holds. And with a cap too small for the lesson, the run fails on the budget before
any review, and nothing is left reserved."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import jwt
import pytest
from sqlalchemy import text

from app.api.app import create_app
from app.api.auth import ROLES, AuthenticationFailed, Principal
from app.config.settings import REPO_ROOT
from tests.auth import drain
from tests.e2e.test_run_cap_api import capped


def load():
    spec = importlib.util.spec_from_file_location("live_smoke", REPO_ROOT / "deploy" / "live_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_smoke"] = module
    spec.loader.exec_module(module)
    return module


live = load()
smoke = live.smoke
bundle = live.review_bundle


class GatewayAuth:
    """The gateway's verification, in process: the script's own tokens (deploy/smoke.py `token`) or 401."""

    def __init__(self, public_key) -> None:
        self.public_key = public_key

    def authenticate(self, request) -> Principal:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthenticationFailed("no credentials")
        try:
            claims = jwt.decode(token, self.public_key, algorithms=["RS256"], audience=smoke.AUDIENCE,
                                issuer=smoke.ISSUER, options={"require": ["exp", "iss", "aud"]})
        except jwt.PyJWTError:
            raise AuthenticationFailed("token rejected") from None
        roles = frozenset(r for r in claims.get("ta_roles", []) if r in ROLES)
        return Principal(user_id=claims["sub"], learner_ids=frozenset({claims["sub"]}), roles=roles)


def rows(container, query: str) -> list[str]:
    """psql -At: one line per row, columns joined by |, NULL as empty."""
    with container._sessions() as s:
        result = s.execute(text(query))
        return ["|".join("" if v is None else str(v) for v in row) for row in result]


@pytest.fixture
def in_process(tmp_path, mock_llm, monkeypatch):
    """`run(cap)` runs live_smoke.main --mock on a capped container (the rehearsal's prices and limits)."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi.testclient import TestClient

    saved_env = dict(os.environ)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    secrets_dir, gateway = tmp_path / "secrets", tmp_path / "gateway"
    gateway.mkdir()
    for module in (live, smoke):
        monkeypatch.setattr(module, "SECRETS", secrets_dir)
    monkeypatch.setattr(smoke, "GATEWAY_CONFIG", gateway)
    monkeypatch.setattr(live, "REPORT", tmp_path / "live-smoke-report.json")
    monkeypatch.setattr(live, "REVIEW_BUNDLE", tmp_path / "live-smoke-review.enc.json")
    monkeypatch.setattr(smoke, "write_secrets", lambda: key)
    monkeypatch.setattr(smoke, "wait_for", lambda *a, **k: None)
    monkeypatch.setattr(live, "video_duration", lambda artifact_id: 12.5)
    monkeypatch.setattr(live, "bucket_leaks", lambda keys: [])
    monkeypatch.setattr(smoke, "bucket_keys", lambda: set())
    monkeypatch.setattr(smoke, "rows_naming", lambda ids: {})
    monkeypatch.setattr(live.subprocess, "run", lambda cmd, **kw: type(
        "Done", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    sealed_for: list = []
    real_recipient = live.review_recipient

    def recipient(*, mock, environ):
        pair = real_recipient(mock=mock, environ=environ)
        sealed_for.append(pair)
        return pair

    monkeypatch.setattr(live, "review_recipient", recipient)
    state: dict = {}

    def run(cap: float, *argv: str) -> int:
        container = capped(tmp_path / f"cap-{cap}", mock_llm, cap)
        state["container"] = container
        app = create_app(container, authenticator=GatewayAuth(key.public_key()))

        class Done:
            returncode, stderr = 0, ""

            def __init__(self, stdout: str = "") -> None:
                self.stdout = stdout

        def compose(*args, check=True, capture=False, stdin=None):
            if args[:3] == ("run", "--rm", "api"):
                return Done(f"run-wide cap: max_cost_usd={cap:g}\nREADY")
            if args[:4] == ("exec", "-T", "gateway", "sh"):
                return Done("/run/secrets:\ngateway_secret\n/run/ta-config:\njwks.json\n")
            if args[:4] == ("exec", "-T", "postgres", "psql"):
                return Done("\n".join(rows(container, stdin)))
            return Done()

        def poll_task(client, task_id, headers, timeout=900):
            drain(container)
            r = client.get(f"/tasks/{task_id}", headers=headers)
            assert r.status_code == 200, r.text
            return r.json()

        monkeypatch.setattr(smoke, "compose", compose)
        monkeypatch.setattr(smoke, "poll_task", poll_task)
        monkeypatch.setattr(smoke.httpx, "Client", lambda **kw: TestClient(app, base_url=kw.get("base_url")))
        return live.main(["--mock", "--max-cost", f"{cap:.2f}", "--no-build", "--deadline-minutes", "5", *argv])

    yield run, state, sealed_for, tmp_path
    if "container" in state:
        state["container"].close()
    os.environ.clear()
    os.environ.update(saved_env)


def report(tmp_path) -> dict:
    return json.loads((tmp_path / "live-smoke-report.json").read_text(encoding="utf-8"))


def test_the_flow_holds_previews_approves_and_seals_the_same_lesson_without_a_paid_review(in_process) -> None:
    run, state, sealed_for, tmp_path = in_process
    assert run(2.0) == 0
    r = report(tmp_path)
    assert r["status"] == "PASSED" and r["leaks"] == [] and r["cleanup_errors"] == []
    container = state["container"]

    # one lesson task, approved through the teacher route by the class's teacher, and read afterwards
    lesson = r["lesson"]
    [task_row] = rows(container, "SELECT task_id FROM tasks;") or [None]
    assert task_row is None or task_row == lesson["task_id"]  # the learner's rows are deleted at the end
    assert lesson["release"] == "APPROVED" and lesson["images"] >= 1
    decided = rows(container, "SELECT status FROM lesson_releases;")
    assert decided in ([], ["APPROVED"])  # erased with the learner, or still the one approval
    lesson_tasks = {e.task_id for e in container.spend.entries() if e.task_id and e.session_id is None}
    assert lesson_tasks == {lesson["task_id"]}  # no second generation was paid for

    # stage times: each stage has a start, an end and a duration, in order
    stages = r["stages"]
    for name in ("stack", "class_setup", "generation", "review", "grade"):
        assert {"started_at", "finished_at", "seconds"} <= set(stages[name]), name
        assert stages[name]["seconds"] >= 0
    order = [stages[n]["started_at"] for n in ("class_setup", "generation", "review", "grade")]
    assert order == sorted(order)
    assert r["generation_server_seconds"] >= 0

    # the lesson's cost comes from the ledger, split by how it is known, and adds up with it
    cost = r["lesson_cost"]
    assert cost["generation"]["attempts"] > 0 and cost["teaching_session"]["attempts"] > 0
    assert cost["charged_usd"] == pytest.approx(cost["measured_usd"] + cost["estimate_usd"], abs=1e-6)
    assert cost["measured_usd"] > 0 and cost["charged_usd"] <= r["spend"]["total_usd"] + 1e-9
    by_task = sum(e.final_cost_microusd or 0 for e in container.spend.entries()
                  if e.task_id == lesson["task_id"] and e.session_id is None
                  and e.state.value in ("settled", "unknown")) / 1_000_000
    assert cost["generation"]["charged_usd"] == pytest.approx(by_task, abs=1e-6)
    assert r["spend"]["total_usd"] <= 2.0 and r["spend"]["entries_still_reserved"] == 0

    # the sealed bundle opens only with the reviewer's key, to the preview, reading, release and images
    assert r["review_bundle"] == "live-smoke-review.enc.json"
    envelope = json.loads((tmp_path / "live-smoke-review.enc.json").read_text(encoding="utf-8"))
    [(public, private)] = sealed_for
    files = bundle.unseal(envelope, private)
    assert {"REVIEW.md", "preview.json", "reading.json", "release.json"} <= set(files)
    images = [n for n in files if n.startswith("images/")]
    assert len(images) == lesson["images"] and all(files[n] for n in images)
    preview, reading = json.loads(files["preview.json"]), json.loads(files["reading.json"])
    release = json.loads(files["release.json"])
    assert preview["task_id"] == reading["task_id"] == release["task_id"] == lesson["task_id"]
    assert preview["lesson_id"] == reading["lesson_id"] == lesson["lesson_id"]
    assert release["status"] == "APPROVED" and "learner_id" not in release and "display_name" not in release
    assert live.answer_key_fields(preview) == set() == live.answer_key_fields(reading)
    sheet = files["REVIEW.md"].decode()
    assert "Human review" in sheet and "lesson_cost" in sheet and preview["sections"][0]["paragraphs"][0] in sheet
    from cryptography.hazmat.primitives.asymmetric import rsa
    stranger = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    with pytest.raises(bundle.BundleError):
        bundle.unseal(envelope, stranger)
    assert "BEGIN" not in (tmp_path / "live-smoke-review.enc.json").read_text()  # no key material in the file


def test_a_failure_after_the_review_still_keeps_the_sealed_lesson(in_process) -> None:
    run, _state, sealed_for, tmp_path = in_process
    assert run(2.0, "--fail-after", "review") == 1
    r = report(tmp_path)
    assert r["status"] == "FAILED" and r["review_bundle"] == "live-smoke-review.enc.json"
    assert "review" in r["stages"] and "grade" not in r["stages"]
    [(_public, private)] = sealed_for
    files = bundle.unseal(json.loads((tmp_path / "live-smoke-review.enc.json").read_text()), private)
    assert json.loads(files["release.json"])["status"] == "APPROVED"


def test_a_cap_too_small_for_the_lesson_fails_before_any_review_and_leaves_nothing_reserved(in_process) -> None:
    run, state, _sealed_for, tmp_path = in_process
    assert run(0.01) == 1
    r = report(tmp_path)
    assert r["status"] == "FAILED" and r["review_bundle"] is None
    assert "review" not in r["stages"]
    assert not (tmp_path / "live-smoke-review.enc.json").exists()
    container = state["container"]
    reserved = rows(container, "SELECT count(*) FROM spend_entries WHERE state = 'reserved';")
    settled = rows(container, "SELECT coalesce(sum(settled), 0) FROM spend_counters "
                              "WHERE scope_key = 'total:all' AND metric = 'cost_microusd';")
    assert reserved == ["0"] and int(settled[0]) <= 10_000  # never past the cap
    assert rows(container, "SELECT count(*) FROM lesson_releases WHERE status = 'APPROVED';") == ["0"]
