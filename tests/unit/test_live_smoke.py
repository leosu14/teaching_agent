"""The live smoke run's own guarantees (deploy/live_smoke.py, MVP item 10), without Docker or keys: configuration
fails closed before anything starts, the ceiling is bounded, the provider keys are found in any form and never
printed, and the cleanup (learner deletion, `docker compose down -v`, key files) runs on success, failure, the
run's deadline, SIGTERM and SIGINT, also when a further signal arrives during it or a step of it fails."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pytest

from app.config.settings import REPO_ROOT

KEY = "sk-live-TEST+key/with=odd_chars-0123456789"


def load():
    spec = importlib.util.spec_from_file_location("live_smoke", REPO_ROOT / "deploy" / "live_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_smoke"] = module
    spec.loader.exec_module(module)
    return module


live = load()
smoke = live.smoke
bundle = live.review_bundle


def reviewer():
    """A reviewer's key pair (3072-bit RSA, the smallest REVIEW_PUBLIC_KEY accepts), made once per test session."""
    global _REVIEWER
    if _REVIEWER is None:
        from cryptography.hazmat.primitives.asymmetric import rsa
        _REVIEWER = rsa.generate_private_key(public_exponent=65537, key_size=bundle.MIN_KEY_BITS)
    return _REVIEWER


_REVIEWER = None
REAL_CLIENT = smoke.httpx.Client  # before any fixture replaces it


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """The script's files under tmp_path, a recording `docker compose`, a fake API, and the signal handlers and
    alarm restored afterwards."""
    secrets_dir, gateway = tmp_path / "secrets", tmp_path / "gateway"
    gateway.mkdir()
    (gateway / "jwks.json").write_text("{}", encoding="utf-8")
    for module in (live, smoke):
        monkeypatch.setattr(module, "SECRETS", secrets_dir)
    monkeypatch.setattr(smoke, "GATEWAY_CONFIG", gateway)
    monkeypatch.setattr(live, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(live, "REVIEW_BUNDLE", tmp_path / "live-smoke-review.enc.json")
    calls: list[tuple[str, ...]] = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def compose(*args, check=True, capture=False, stdin=None):
        calls.append(args)
        if args[:1] == ("logs",):
            done = Done()
            done.stdout = f"api | something with {KEY} in it\n"
            return done
        return Done()

    monkeypatch.setattr(smoke, "compose", compose)
    deletes: list[str] = []

    class Client:
        def __init__(self, **_kw) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def delete(self, path, headers=None):
            deletes.append(path)
            return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(smoke.httpx, "Client", Client)
    docker: list[list[str]] = []

    def run_docker(cmd, **_kw):
        docker.append(cmd)
        return type("Done", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(live.subprocess, "run", run_docker)
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)}
    yield type("Env", (), {"calls": calls, "deletes": deletes, "docker": docker, "secrets": secrets_dir, "gateway": gateway,
                           "report": tmp_path / "report.json", "bundle": tmp_path / "live-smoke-review.enc.json",
                           "use": staticmethod(lambda flow: monkeypatch.setattr(live, "flow", flow))})
    signal.alarm(0)
    for s, handler in saved.items():
        signal.signal(s, handler)


def started_flow(behaviour):
    """A flow that brings the stack up and creates the learner (as the real one does), then `behaviour`."""
    def flow(run, _environment):
        run.started = True
        run.learner, run.learner_headers = "live-abc", {"Authorization": "Bearer x"}
        behaviour(run)
    return flow


def down_called(calls) -> bool:
    return any(c[:1] == ("down",) and "-v" in c and "--remove-orphans" in c for c in calls)


# --- configuration ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-1", "5.01", "nan", "inf", "abc", ""])
def test_the_ceiling_must_be_positive_finite_and_bounded(value: str) -> None:
    with pytest.raises(live.ConfigProblem):
        live.ceiling(value)
    assert live.ceiling("1.00") == 1.0 and live.ceiling("5") == 5.0


LIVE_ENV = {"LLM_MODEL": "gpt-x", "LLM_INPUT_PRICE_PER_MTOK": "2.5", "LLM_OUTPUT_PRICE_PER_MTOK": "10",
            "TTS_PRICE_PER_MCHAR": "15", "IMAGE_PRICE_PER_IMAGE": "0.04", "SEARCH_PRICE_PER_REQUEST": "0.008",
            "TA_VENDOR_CAPS": "openai=50,tavily=10", "TTS_MODEL": "tts-1", "IMAGE_MODEL": "gpt-image-1",
            "IMAGE_QUALITY": "low", "IMAGE_SIZE": "1024x1024", "IMAGE_INPUT_PRICE_PER_MTOK": "5"}


def test_a_complete_live_configuration_is_passed_to_the_stack() -> None:
    env = live.stack_environment(mock=False, max_cost=1.0, environ=LIVE_ENV)
    assert env["TEACHING_AGENT_MODE"] == "production" and env["TA_BUDGET_TOTAL_MAX_COST_USD"] == "1"
    assert (env["LLM_PROVIDER"], env["TTS_PROVIDER"], env["IMAGE_PROVIDER"], env["SEARCH_PROVIDER"]) == (
        "openai", "openai", "openai", "tavily")
    assert env["TA_COURSES"] == "spanish-a1-b1" and env["TTS_PRICE_PER_MCHAR"] == "15"
    assert not any("KEY" in name and "FILE" not in name for name in env)  # no key ever goes in the environment


@pytest.mark.parametrize(("change", "problem"), [
    ({"TTS_PRICE_PER_MCHAR": ""}, "TTS_PRICE_PER_MCHAR is not set"),
    ({"IMAGE_PRICE_PER_IMAGE": "free"}, "is not a number"),
    ({"SEARCH_PRICE_PER_REQUEST": "0"}, "must be a positive price"),
    ({"LLM_INPUT_PRICE_PER_MTOK": "inf"}, "must be a positive price"),
    ({"LLM_OUTPUT_PRICE_PER_MTOK": "-2"}, "must be a positive price"),
    ({"LLM_MODEL": ""}, "LLM_MODEL is not set"),
    ({"TA_VENDOR_CAPS": "openai=50"}, "tavily console"),
    ({"TTS_MODEL": ""}, "TTS_MODEL is not set"),
    ({"IMAGE_QUALITY": ""}, "IMAGE_QUALITY is not set"),
    ({"IMAGE_SIZE": " "}, "IMAGE_SIZE is not set"),
    ({"IMAGE_INPUT_PRICE_PER_MTOK": ""}, "bills the prompt's input tokens"),
])
def test_a_missing_or_invalid_price_model_or_vendor_cap_fails_closed(change: dict, problem: str) -> None:
    with pytest.raises(live.ConfigProblem, match=problem):
        live.stack_environment(mock=False, max_cost=1.0, environ={**LIVE_ENV, **change})


def test_the_rehearsal_is_offline_and_prices_every_paid_capability() -> None:
    env = live.stack_environment(mock=True, max_cost=2.0, environ={})
    assert env["TEACHING_AGENT_MODE"] == "offline"
    assert {env["TTS_PROVIDER"], env["IMAGE_PROVIDER"], env["SEARCH_PROVIDER"]} == {"mock"}
    assert all(float(env[v]) > 0 for v in ("TTS_PRICE_PER_MCHAR", "IMAGE_PRICE_PER_IMAGE",
                                            "SEARCH_PRICE_PER_REQUEST"))


def test_a_configuration_problem_starts_nothing(isolated) -> None:
    assert live.main(["--max-cost", "9"]) == 2
    assert live.main(["--max-cost", "1"]) == 2  # no prices in this environment
    assert not [c for c in isolated.calls if c[:1] == ("up",)] and not down_called(isolated.calls)


def test_live_keys_must_be_present_and_the_rehearsal_writes_fake_ones(isolated) -> None:
    with pytest.raises(live.ConfigProblem, match="openai_api_key is missing"):
        live.Keys.load(mock=False)
    keys = live.Keys.load(mock=True)
    assert set(keys.values) == {"OPENAI_API_KEY", "TAVILY_API_KEY"}
    assert all(v.startswith("fake-") for v in keys.values.values())
    assert (isolated.secrets / "openai_api_key").read_text() == keys.values["OPENAI_API_KEY"]


# --- the keys: found in any form, never printed ------------------------------------------------------------------------


@pytest.mark.parametrize("encode", [
    lambda k: k,
    lambda k: base64.b64encode(k.encode()).decode(),
    lambda k: base64.urlsafe_b64encode(k.encode()).decode().rstrip("="),
    lambda k: quote(k, safe=""),
])
def test_a_key_is_found_raw_base64_or_url_encoded_and_redacted(encode) -> None:
    keys = live.Keys({"OPENAI_API_KEY": KEY})
    text = f"log line: Authorization={encode(KEY)} done"
    assert keys.scan(text) == ["OPENAI_API_KEY"]
    redacted = keys.redact(text)
    assert keys.scan(redacted) == [] and "***" in redacted
    assert keys.scan("nothing secret here") == []


def test_a_key_in_a_report_is_never_written(isolated) -> None:
    run = live.Run(mock=True, max_cost=1.0, keys=live.Keys({"OPENAI_API_KEY": KEY}))
    run.report = {"status": "PASSED", "oops": KEY}
    live.write_report(run)
    assert not isolated.report.exists() and run.leaks == ["a key in the report (not written)"]
    run.report, run.leaks = {"status": "PASSED", "total_usd": 0.12}, []
    live.write_report(run)
    assert json.loads(isolated.report.read_text())["total_usd"] == 0.12


def test_answer_key_fields_are_found_anywhere_in_a_response() -> None:
    assert live.answer_key_fields({"a": [{"expected_answer": "x"}], "b": {"c": {"accepted_answers": []}}}) == {
        "expected_answer", "accepted_answers"}
    assert live.answer_key_fields({"waiting_question": {"prompt": "¿Cómo te llamas?", "choices": []}}) == set()


# --- cleanup on every way a run can end -------------------------------------------------------------------------------


def test_a_failed_run_deletes_the_learner_takes_the_stack_down_and_never_prints_a_key(isolated, capsys) -> None:
    def boom(run):
        run.keys = live.Keys({"OPENAI_API_KEY": KEY})
        raise RuntimeError(f"provider said no to {KEY}")

    isolated.use(started_flow(boom))
    assert live.main(["--mock", "--max-cost", "1"]) == 1
    out = capsys.readouterr()
    assert KEY not in out.out + out.err and "***" in out.err  # the message and the failure logs are redacted
    assert isolated.deletes == ["/learners/live-abc"] and down_called(isolated.calls)
    assert not isolated.secrets.exists() and not (isolated.gateway / "jwks.json").exists()
    report = json.loads(isolated.report.read_text())
    assert report["status"] == "FAILED" and KEY not in isolated.report.read_text()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_cancelled_run_cleans_up(isolated, sig) -> None:
    def cancelled(_run):
        os.kill(os.getpid(), sig)
        time.sleep(5)  # interrupted by the handler
        raise AssertionError("the signal did not interrupt the run")

    isolated.use(started_flow(cancelled))
    assert live.main(["--mock", "--max-cost", "1"]) == 1
    assert isolated.deletes and down_called(isolated.calls) and not isolated.secrets.exists()
    assert json.loads(isolated.report.read_text())["failure"] == "Interrupted"


def test_the_runs_own_deadline_interrupts_it_and_cleans_up(isolated) -> None:
    isolated.use(started_flow(lambda _run: time.sleep(5)))
    started = time.monotonic()
    assert live.main(["--mock", "--max-cost", "1", "--deadline-minutes", "0.01"]) == 1
    assert time.monotonic() - started < 4.5
    assert down_called(isolated.calls) and not isolated.secrets.exists()


def test_a_signal_during_cleanup_does_not_stop_it_and_a_failing_step_does_not_skip_the_rest(isolated,
                                                                                         monkeypatch) -> None:
    recorded = []

    def compose(*args, check=True, capture=False, stdin=None):
        recorded.append(args)
        if args[:1] == ("down",):
            os.kill(os.getpid(), signal.SIGTERM)  # a second cancellation while the stack goes down: ignored
            raise OSError("docker went away")
        return type("Done", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(smoke, "compose", compose)

    def fail(_run):
        raise RuntimeError("the lesson failed")

    isolated.use(started_flow(fail))
    assert live.main(["--mock", "--max-cost", "1"]) == 1
    assert any(a[:1] == ("down",) for a in recorded)
    assert not isolated.secrets.exists()  # removed although `down` failed
    assert "docker compose down failed (OSError)" in json.loads(isolated.report.read_text())["cleanup_errors"]


def test_a_successful_run_still_cleans_up_and_a_leak_fails_it(isolated) -> None:
    def passes(run):
        run.report = {"status": "PASSED"}

    isolated.use(started_flow(passes))
    assert live.main(["--mock", "--max-cost", "1"]) == 0
    assert down_called(isolated.calls) and not isolated.secrets.exists() and isolated.deletes

    def leaks(run):
        run.report = {"status": "PASSED"}
        run.leaks.append("OPENAI_API_KEY in the service logs (before the deletion)")

    isolated.use(started_flow(leaks))
    assert live.main(["--mock", "--max-cost", "1"]) == 1
    assert json.loads(isolated.report.read_text())["leaks"] == [
        "OPENAI_API_KEY in the service logs (before the deletion)"]


def test_the_forced_failure_stages_raise() -> None:
    run = live.Run(mock=True, max_cost=1.0, fail_after="lesson")
    live.stop_if(run, "stack")
    with pytest.raises(RuntimeError, match="forced failure after the lesson stage"):
        live.stop_if(run, "lesson")


def test_verify_clean_names_what_is_left(isolated, monkeypatch, capsys) -> None:
    class Listed:
        def __init__(self, out: str) -> None:
            self.returncode, self.stdout, self.stderr = 0, out, ""

    outputs = {"ps": "abc123\n", "volume": "", "network": ""}
    monkeypatch.setattr(live.subprocess, "run",
                        lambda cmd, **_kw: Listed(outputs["ps" if cmd[1] == "ps" else cmd[1]]))
    isolated.secrets.mkdir()
    assert live.verify_clean() == 1
    err = capsys.readouterr().err
    assert "1 container(s)" in err and "secrets" in err and "jwks.json" in err
    outputs["ps"] = ""
    isolated.secrets.rmdir()
    (isolated.gateway / "jwks.json").unlink()
    assert live.verify_clean() == 0


def test_teardown_removes_the_projects_containers_volumes_networks_and_key_files(isolated, monkeypatch) -> None:
    removed: list[list[str]] = []

    def docker(cmd, **_kw):
        listing = cmd[1] in ("ps", "volume", "network") and ("ls" in cmd or cmd[1] == "ps")
        removed.append(cmd)
        out = {"ps": "c1 c2", "volume": "v1", "network": "n1"}[cmd[1]] if listing and "rm" not in cmd else ""
        return type("Done", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(live.subprocess, "run", docker)
    isolated.secrets.mkdir()
    (isolated.secrets / "openai_api_key").write_text("x")
    assert live.main(["teardown"]) == 0
    assert ["docker", "rm", "-f", "c1", "c2"] in removed and ["docker", "volume", "rm", "-f", "v1"] in removed
    assert ["docker", "network", "rm", "n1"] in removed
    assert all("label=com.docker.compose.project=ta-live" in c for c in removed if "--filter" in c)
    assert not isolated.secrets.exists() and not (isolated.gateway / "jwks.json").exists()


def test_the_ledger_assertions_hold_the_ceiling() -> None:
    good = {"entries_still_reserved": 0, "attempts_outside_the_cap": 0, "unpriced_attempts": 0, "total_usd": 0.4,
            "ceiling_usd": 1.0, "by_capability": {c: {"cost_usd": 0.1, "sources": {"priced": 3}}
                                                  for c in ("llm", "tts", "image", "search")}}
    live.assert_ledger(good, mock=True)
    unreconciled = {**good["by_capability"], "llm": {"cost_usd": 0.1, "sources": {"estimate": 3, "priced": 1}}}
    for change in ({"total_usd": 1.01}, {"entries_still_reserved": 1}, {"attempts_outside_the_cap": 2},
                   {"unpriced_attempts": 1}, {"total_usd": 0.0}, {"by_capability": {"llm": {"cost_usd": 1}}},
                   {"by_capability": unreconciled}):
        with pytest.raises(AssertionError):
            live.assert_ledger({**good, **change}, mock=True)


def test_the_compose_command_adds_the_live_override_under_its_own_project() -> None:
    cmd = live.compose_command()
    assert cmd[-2:] == ["-p", "ta-live"] and str(Path("deploy") / "compose.live.yaml") in " ".join(cmd)


# --- B1: the identity tokens outlive no request -------------------------------------------------------------------------


class FakeClock:
    """The IdP's and the polling's clock (deploy/smoke.py's `time`): `sleep` advances it, nothing waits."""

    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Gateway:
    """The gateway and API as the live flow sees them, verifying every token against the fake clock as the real
    gateway does (signature, issuer, audience, `exp` with 30 s leeway): an expired token gets 401. The lesson runs
    for 20 minutes before its diagnostic and 15 more after it, so it outlives any one token (10 minutes)."""

    LEEWAY = 30

    def __init__(self, key, clock: FakeClock) -> None:
        self.public = key.public_key()
        self.clock = clock
        self.lid: str | None = None
        self.phase, self.until = "QUEUED", 0.0
        self.deleted = False
        self.rejected: list[str] = []
        self.subjects: dict[str, int] = {}
        self.approval = self.joined = False
        self.release: str | None = None  # the lesson's release: PENDING once requested in the approving class
        self.order: list[str] = []  # the lesson's surfaces in the order they were reached

    def subject(self, request) -> str | None:
        import jwt
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            return None
        try:
            claims = jwt.decode(header[7:], self.public, algorithms=["RS256"], audience=smoke.AUDIENCE,
                                issuer=smoke.ISSUER, options={"verify_exp": False, "verify_iat": False})
        except jwt.PyJWTError:
            return None
        if claims["exp"] + self.LEEWAY < self.clock.now:
            self.rejected.append(f"{request.method} {request.url.path}")
            return None
        self.subjects[claims["sub"]] = self.subjects.get(claims["sub"], 0) + 1
        return claims["sub"]

    def task(self) -> dict:
        if self.phase == "RUNNING_1" and self.clock.now >= self.until:
            self.phase = "WAITING"
        if self.phase == "RUNNING_2" and self.clock.now >= self.until:
            self.phase = "COMPLETED"
        status = {"RUNNING_1": "RUNNING", "RUNNING_2": "RUNNING"}.get(self.phase, self.phase)
        body = {"task_id": "task_1", "status": status, "release": self.release,
                "created_at": "2027-01-15T08:00:00Z", "updated_at": "2027-01-15T08:35:00Z",
                "job": {"status": "RUNNING" if status == "RUNNING" else "SUCCEEDED"}}
        if status == "WAITING":
            body["waiting"] = {"prompt": {"questions": [{"question_id": "q1", "prompt": "¿Cómo te llamas?"}]}}
        return body

    def __call__(self, request):
        import httpx
        sub, path, method = self.subject(request), request.url.path, request.method

        def answer(status: int, body=None):
            return httpx.Response(status, json=body if body is not None else {})

        if sub is None:
            return answer(401, {"detail": "unauthenticated"})
        me = sub == self.lid
        held = self.release == "PENDING"
        teacher = sub == "teacher-1"
        reading = {"lesson_id": "art_l", "task_id": "task_1", "title": "Saludos", "level": "A1",
                   "sections": [{"section_id": "s1", "purpose": "explanation", "paragraphs": ["Hola means hello."],
                                 "images": [{"artifact_id": "art_i"}]}], "withheld": 1}
        if method == "POST" and path == "/classes" and teacher:
            return answer(201, {"class_id": "cls_1", "join_code": "JOIN-1", "lesson_approval": False})
        if method == "PATCH" and path == "/classes/cls_1" and teacher:
            self.approval = True
            return answer(200, {"class_id": "cls_1", "lesson_approval": True})
        if method == "POST" and path == f"/learners/{self.lid}/classes" and me:
            self.joined = True
            return answer(201, {"class_id": "cls_1", "learner_id": self.lid})
        if path == "/classes/cls_1/lessons" and teacher:
            return answer(200, [{"task_id": "task_1", "status": self.release}] if held else [])
        if path == "/classes/cls_1/lessons/task_1/preview":
            if not teacher:
                return answer(404 if sub == "teacher-2" else 403)
            self.order.append("preview")
            return answer(200, reading)
        if method == "POST" and path == "/classes/cls_1/lessons/task_1/approve" and teacher and held:
            self.release = "APPROVED"
            self.order.append("approve")
            return answer(200, {"task_id": "task_1", "lesson_id": "art_l", "learner_id": self.lid,
                                "display_name": "Live smoke learner", "status": "APPROVED",
                                "decided_at": "2027-01-15T08:36:00Z", "note": "ok"})
        if method == "POST" and path == "/classes/cls_1/lessons/task_1/reject":
            return answer(409)
        if path == "/lessons/task_1/reading" and me:
            if held:
                return answer(409)
            self.order.append("reading")
            return answer(200, reading)
        if path == f"/learners/{self.lid}/progress" and me:
            return answer(200, {"lessons_completed": 1})
        if path == "/classes/cls_1/dashboard" and teacher:
            return answer(200, {"learners": [{"learner_id": self.lid}]})
        if method == "GET" and path == "/artifacts/art_i/content" and teacher:
            return httpx.Response(200, content=b"png", headers={"content-type": "image/png"})
        if held and path in ("/artifacts/art_i/content", "/lessons/task_1/teaching-session"):
            return answer(409)
        if method == "PUT" and path.startswith("/learners/"):
            self.lid = path.rsplit("/", 1)[1]
            return answer(200, {"learner_id": self.lid})
        if method == "GET" and path.startswith("/learners/"):
            return answer(200 if me and not self.deleted else 404)
        if method == "POST" and path == "/tasks":
            self.phase, self.until = "RUNNING_1", self.clock.now + 20 * 60
            self.release = "PENDING" if self.approval and self.joined else None
            return answer(202, {"task_id": "task_1", "status": "QUEUED", "job": {"status": "QUEUED"}})
        if method == "POST" and path == "/tasks/task_1/answers":
            self.phase, self.until = "RUNNING_2", self.clock.now + 15 * 60
            return answer(202, {"task_id": "task_1", "status": "QUEUED", "job": {"status": "QUEUED"}})
        if method == "GET" and path == "/tasks/task_1":
            return answer(200, self.task()) if me else answer(404)
        if method == "GET" and path == "/tasks/task_1/artifacts":
            return answer(200, [{"artifact_id": "art_v", "type": "VIDEO"},
                                {"artifact_id": "art_i", "type": "IMAGE_ASSET"},
                                {"artifact_id": "art_w", "type": "IMAGE_ASSET"}])  # withheld by the content policy
        if path == "/artifacts/art_w/content":
            return answer(409 if held and me else 404)  # refused to everyone, the teacher included
        if method == "GET" and path == "/artifacts/art_i/content":
            return httpx.Response(200 if me else 404, content=b"png")
        if method == "GET" and path == "/artifacts/art_v/content":
            return answer(404)
        if method == "POST" and path == "/lessons/task_1/teaching-session":
            self.clock.sleep(4 * 60)  # the teacher's first turn
            return answer(201, {"session_id": "ts_1", "waiting_question": {"choices": ["Me llamo Ana."]}})
        if method == "POST" and path == "/teaching-sessions/ts_1/answers":
            self.clock.sleep(4 * 60)  # grading
            return answer(200, {"answer_outcome": "correct", "correct": True})
        if method == "DELETE" and path.startswith("/learners/"):
            if sub == "teacher-1":
                return answer(403)
            if not me:
                return answer(404)
            self.deleted = True
            return answer(200, {"objects_pending": 0})
        return answer(404)


@pytest.fixture
def long_lesson(isolated, monkeypatch):
    """The real `flow` against the fake gateway, on the fake clock, with Docker, SQL and the scans stubbed."""
    import httpx
    from cryptography.hazmat.primitives.asymmetric import rsa

    clock = FakeClock()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    gateway = Gateway(key, clock)
    monkeypatch.setattr(smoke, "time", clock)
    monkeypatch.setattr(smoke, "write_secrets", lambda: key)
    monkeypatch.setattr(smoke, "wait_for", lambda *a, **k: None)
    monkeypatch.setattr(smoke.httpx, "Client", lambda **kw: REAL_CLIENT(
        transport=httpx.MockTransport(gateway), base_url=kw.get("base_url", "http://gw"), timeout=5))

    class Done:
        returncode, stderr = 0, ""

        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    def compose(*args, check=True, capture=False, stdin=None):
        if args[:3] == ("run", "--rm", "api"):
            return Done("run-wide cap: max_cost_usd=1\nREADY")
        if args[:4] == ("exec", "-T", "gateway", "sh"):
            return Done("/run/secrets:\ngateway_secret\n/run/ta-config:\njwks.json\n")
        return Done()

    monkeypatch.setattr(smoke, "compose", compose)
    monkeypatch.setattr(smoke, "sql", lambda q: ["task_1"] if "SELECT task_id" in q else
                        ["0"] if "count(*)" in q and "GROUP BY" not in q else [])
    monkeypatch.setattr(smoke, "rows_naming", lambda ids: {})
    monkeypatch.setattr(smoke, "bucket_keys", lambda: set())
    monkeypatch.setattr(live, "video_duration", lambda artifact_id: 100.0)
    monkeypatch.setattr(live, "scan_everything", lambda run, when: None)
    monkeypatch.setattr(live, "ledger", lambda max_cost: {"total_microusd": 1})
    monkeypatch.setattr(live, "assert_ledger", lambda spend, mock: None)
    return clock, gateway


def test_a_lesson_longer_than_a_token_completes_with_every_request_authenticated(long_lesson) -> None:
    clock, gateway = long_lesson
    start = clock.now
    run = live.Run(mock=False, max_cost=1.0, build=False, keys=live.Keys({"OPENAI_API_KEY": KEY}),
                   review_key=reviewer().public_key())
    live.flow(run, {})
    assert clock.now - start > 40 * 60  # the run outlived four token lifetimes
    assert gateway.rejected == []  # no request carried an expired token
    assert run.deleted and gateway.deleted  # the deletion through the API still worked at the end
    assert run.report["status"] == "PASSED" and run.report["graded_outcome"] == "correct"
    assert gateway.subjects[gateway.lid] > 100  # every poll signed its own token
    # the lesson was held, previewed and approved by its teacher before the learner read it
    assert gateway.order == ["preview", "approve", "reading"] and gateway.release == "APPROVED"
    assert gateway.subjects["teacher-1"] >= 5 and gateway.subjects["teacher-2"] == 1  # refused another teacher
    stages = run.report["stages"]
    assert stages["generation"]["seconds"] >= 35 * 60  # the fake lesson's 35 minutes, on the run's clock
    assert stages["grade"]["seconds"] >= 8 * 60 and stages["review"]["seconds"] >= 0
    assert run.report["generation_server_seconds"] == 35 * 60  # created_at to updated_at of the completed task
    assert run.bundle == "live-smoke-review.enc.json"
    # an image the content policy withholds (404 even for the teacher) is counted, not a failure
    assert run.report["lesson"]["images"] == 2 and run.report["lesson"]["images_withheld"] == 1


def test_cleanup_deletes_the_learner_with_a_fresh_token_long_after_the_lesson_began(long_lesson) -> None:
    clock, gateway = long_lesson
    run = live.Run(mock=False, max_cost=1.0, build=False, keys=live.Keys({"OPENAI_API_KEY": KEY}),
                   review_key=reviewer().public_key())
    run.fail_after = "lesson"
    with pytest.raises(RuntimeError, match="forced failure"):
        live.flow(run, {})
    assert not gateway.deleted
    clock.sleep(30 * 60)  # e.g. the deadline approaching
    live.cleanup(run)
    assert gateway.rejected == [] and gateway.deleted and run.deleted, run.cleanup_errors
    assert run.cleanup_errors == []


def test_the_fake_gateway_refuses_a_token_minted_once_as_before(long_lesson) -> None:
    """The regression the two tests above guard: one token minted at the start is refused 10.5 minutes later."""
    import httpx
    clock, gateway = long_lesson
    key = smoke.write_secrets()
    stale = {"Authorization": f"Bearer {smoke.token(key, 'live-x')}"}
    with smoke.httpx.Client(base_url="http://gw") as c:
        assert c.put("/learners/live-x", json={}, headers=stale).status_code == 200
        clock.sleep(11 * 60)
        assert c.get("/learners/live-x", headers=stale).status_code == 401
        assert c.get("/learners/live-x", headers=live.Bearer(key, "live-x")).status_code == 200
    assert gateway.rejected == ["GET /learners/live-x"]
    assert isinstance(httpx.Headers(live.Bearer(key, "a")), httpx.Headers)


# --- the review bundle: the lesson, sealed for the human reviewer ---------------------------------------------------


def pem(public_key) -> str:
    from cryptography.hazmat.primitives import serialization
    return public_key.public_bytes(serialization.Encoding.PEM,
                                   serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def test_the_live_run_needs_the_reviewers_rsa_public_key_and_checks_it_before_anything_starts() -> None:
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    public, private = live.review_recipient(mock=False, environ={"REVIEW_PUBLIC_KEY": pem(reviewer().public_key())})
    assert private is None and bundle.fingerprint(public) == bundle.fingerprint(reviewer().public_key())
    escaped = pem(reviewer().public_key()).replace("\n", "\\n")  # a one-line variable
    assert live.review_recipient(mock=False, environ={"REVIEW_PUBLIC_KEY": escaped})[0].key_size == 3072
    weak = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    curve = ec.generate_private_key(ec.SECP256R1()).public_key()
    for environ, problem in (({}, "REVIEW_PUBLIC_KEY is not set"),
                             ({"REVIEW_PUBLIC_KEY": "not a key"}, "not a PEM public key"),
                             ({"REVIEW_PUBLIC_KEY": pem(weak)}, "at least 3072 bits"),
                             ({"REVIEW_PUBLIC_KEY": pem(curve)}, "must be an RSA public key")):
        with pytest.raises(live.ConfigProblem, match=problem):
            live.review_recipient(mock=False, environ=environ)
    public, private = live.review_recipient(mock=True, environ={})  # the rehearsal's own throwaway pair
    assert private is not None and bundle.fingerprint(private.public_key()) == bundle.fingerprint(public)


def test_a_live_run_without_the_review_key_starts_nothing(isolated, monkeypatch) -> None:
    prices = {"LLM_MODEL": "gpt-x", "LLM_INPUT_PRICE_PER_MTOK": "0.4", "LLM_OUTPUT_PRICE_PER_MTOK": "1.6",
              "TTS_PRICE_PER_MCHAR": "15", "IMAGE_PRICE_PER_IMAGE": "0.005", "SEARCH_PRICE_PER_REQUEST": "0.008",
              "TTS_MODEL": "tts-1", "IMAGE_MODEL": "gpt-image-1-mini", "IMAGE_QUALITY": "low",
              "IMAGE_SIZE": "1024x1024", "IMAGE_INPUT_PRICE_PER_MTOK": "2", "TA_VENDOR_CAPS": "openai=5,tavily=5"}
    for name, value in prices.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("REVIEW_PUBLIC_KEY", raising=False)
    assert live.main(["--max-cost", "1.00"]) == 2
    assert isolated.calls == [] and not isolated.bundle.exists()


def test_a_sealed_bundle_opens_only_with_the_reviewers_key_and_unaltered() -> None:
    from cryptography.hazmat.primitives.asymmetric import rsa

    files = {"REVIEW.md": "# Hola".encode(), "images/art_1.png": b"\x89PNG..."}
    envelope = bundle.seal(files, reviewer().public_key())
    assert bundle.unseal(envelope, reviewer()) == files
    assert b"Hola" not in json.dumps(envelope).encode()
    stranger = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    with pytest.raises(bundle.BundleError, match="another key"):
        bundle.unseal(envelope, stranger)
    with pytest.raises(bundle.BundleError, match="cannot be decrypted"):  # relabelled for the stranger: refused
        bundle.unseal({**envelope, "recipient_sha256": bundle.fingerprint(stranger.public_key())}, stranger)
    flipped = base64.b64decode(envelope["ciphertext"])
    flipped = base64.b64encode(flipped[:-1] + bytes([flipped[-1] ^ 1])).decode()
    with pytest.raises(bundle.BundleError, match="cannot be decrypted"):
        bundle.unseal({**envelope, "ciphertext": flipped}, reviewer())
    with pytest.raises(bundle.BundleError, match="not a"):
        bundle.unseal({**envelope, "format": "other"}, reviewer())
    for name in ("../escape.md", "/abs.md", "C:/x.md"):
        with pytest.raises(bundle.BundleError, match="unsafe file name"):
            bundle.pack({name: b"x"})


def test_the_reviewer_opens_the_bundle_into_an_empty_folder(tmp_path, capsys) -> None:
    from cryptography.hazmat.primitives import serialization

    key_file = tmp_path / "review.pem"
    key_file.write_bytes(reviewer().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
    sealed = tmp_path / "live-smoke-review.enc.json"
    sealed.write_text(json.dumps(bundle.seal({"REVIEW.md": b"# Hola", "images/a.png": b"png"},
                                             reviewer().public_key())))
    out = tmp_path / "review"
    assert bundle.main(["open", str(sealed), "--private-key", str(key_file), "--out", str(out)]) == 0
    assert (out / "REVIEW.md").read_bytes() == b"# Hola" and (out / "images" / "a.png").read_bytes() == b"png"
    assert bundle.main(["open", str(sealed), "--private-key", str(key_file), "--out", str(out)]) == 1  # not empty
    assert "not empty" in capsys.readouterr().err


def test_a_provider_key_in_the_lesson_is_a_leak_and_no_bundle_is_written(isolated) -> None:
    run = live.Run(mock=True, max_cost=1.0, keys=live.Keys({"OPENAI_API_KEY": KEY}),
                   review_key=reviewer().public_key(), review_private_key=reviewer())
    live.write_review_bundle(run, {"preview.json": json.dumps({"p": f"text {KEY}"}).encode()})
    assert run.leaks == ["OPENAI_API_KEY in the review bundle file preview.json (not written)"]
    assert run.bundle is None and not isolated.bundle.exists()
    run.leaks = []
    live.write_review_bundle(run, {"preview.json": b'{"p": "clean"}'})  # the rehearsal checks it opens again
    assert run.bundle == isolated.bundle.name and isolated.bundle.exists()


def test_teardown_and_verify_clean_keep_the_sealed_bundle_for_the_upload(isolated, monkeypatch, capsys) -> None:
    monkeypatch.setattr(live, "ROOT", isolated.bundle.parent)
    isolated.bundle.write_text("{}", encoding="utf-8")
    assert live.teardown() == 0 and isolated.bundle.exists()
    assert live.verify_clean() == 0 and "left behind" not in capsys.readouterr().err


def test_the_lessons_cost_is_split_into_measured_and_estimate_from_the_ledger(monkeypatch) -> None:
    queries: list[str] = []

    def sql(query: str) -> list[str]:
        queries.append(query)
        if "state = 'failed'" in query:
            return ["1"]
        if "task_id = 'task_1'" in query:
            return ["estimate|1|20000", "priced|10|250000", "reported|1|5000"]
        if "session_id = 'ts_1'" in query:
            return ["priced|2|6000"]
        return []

    monkeypatch.setattr(smoke, "sql", sql)
    cost = live.lesson_cost("task_1", "ts_1")
    assert cost["generation"] == {"attempts": 12, "charged_usd": 0.275, "measured_usd": 0.255, "estimate_usd": 0.02,
                                  "failed_attempts": 1, "sources": {"estimate": 1, "priced": 10, "reported": 1}}
    assert cost["teaching_session"]["measured_usd"] == 0.006
    assert (cost["charged_usd"], cost["measured_usd"], cost["estimate_usd"]) == (0.281, 0.261, 0.02)
    assert "session_id IS NULL" in queries[0]  # the session's attempts are not counted twice
    assert all("state IN ('settled', 'unknown')" in q or "state = 'failed'" in q for q in queries)


# --- the workflows: the paid run only by hand, on main, in `staging`; CI never reaches a paid provider -------------


def workflow(name: str) -> str:
    return (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_the_live_workflow_keeps_its_one_dollar_default_and_uploads_the_sealed_lesson() -> None:
    text = workflow("live-smoke.yml")
    trigger = text.split("on:", 1)[1].split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger and not any(t in trigger for t in ("push", "pull_request", "schedule"))
    assert 'default: "1.00"' in text and "environment: staging" in text
    assert "if: github.ref == 'refs/heads/main'" in text
    assert "REVIEW_PUBLIC_KEY: ${{ vars.REVIEW_PUBLIC_KEY }}" in text  # a variable: a public key is not a secret
    upload = text.split("name: live-smoke-review", 1)[1].split("\n      - ", 1)[0]
    assert "path: live-smoke-review.enc.json" in upload and "retention-days: 14" in upload
    assert text.index("live_smoke.py teardown") < text.index("name: live-smoke-review")
    assert live.MAX_CEILING_USD == 5.00 and live.ceiling("1.00") == 1.0


def test_ci_rehearses_the_review_stage_on_mock_providers_without_any_secret() -> None:
    text = workflow("ci.yml")
    assert "secrets." not in text and "environment:" not in text  # no provider key can reach a CI job
    assert "live_smoke.py --mock --max-cost 2.00" in text
    assert "--fail-after review" in text  # a failure after the teacher's approval is rehearsed, and cleaned up
    assert "test -s live-smoke-review.enc.json" in text  # ... and must still have kept the sealed lesson


@pytest.mark.parametrize("stage, at", [("class_setup", 1), ("review", 3)])
def test_a_paid_call_during_the_class_setup_or_the_review_fails_the_run(long_lesson, monkeypatch, stage, at) -> None:
    """The ledger's attempt count is read before and after each stage that must not pay: a change fails the run."""
    reads = iter(range(100))
    counts = []

    def paid_attempts():
        n = next(reads)
        counts.append(n)
        return 0 if n < at else 1  # an attempt appears between the stage's two reads

    monkeypatch.setattr(live, "paid_attempts", paid_attempts)
    run = live.Run(mock=False, max_cost=1.0, build=False, keys=live.Keys({"OPENAI_API_KEY": KEY}),
                   review_key=reviewer().public_key())
    with pytest.raises(AssertionError, match="made a paid provider call"):
        live.flow(run, {})
    assert stage in run.stages and "finished_at" not in run.stages[stage]
