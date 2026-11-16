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
                           "report": tmp_path / "report.json",
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
        body = {"task_id": "task_1", "status": status,
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
        if method == "PUT" and path.startswith("/learners/"):
            self.lid = path.rsplit("/", 1)[1]
            return answer(200, {"learner_id": self.lid})
        if method == "GET" and path.startswith("/learners/"):
            return answer(200 if me and not self.deleted else 404)
        if method == "POST" and path == "/tasks":
            self.phase, self.until = "RUNNING_1", self.clock.now + 20 * 60
            return answer(202, {"task_id": "task_1", "status": "QUEUED", "job": {"status": "QUEUED"}})
        if method == "POST" and path == "/tasks/task_1/answers":
            self.phase, self.until = "RUNNING_2", self.clock.now + 15 * 60
            return answer(202, {"task_id": "task_1", "status": "QUEUED", "job": {"status": "QUEUED"}})
        if method == "GET" and path == "/tasks/task_1":
            return answer(200, self.task()) if me else answer(404)
        if method == "GET" and path == "/tasks/task_1/artifacts":
            return answer(200, [{"artifact_id": "art_v", "type": "VIDEO"},
                                {"artifact_id": "art_i", "type": "IMAGE_ASSET"}])
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
    monkeypatch.setattr(smoke, "sql", lambda q: ["task_1"] if "SELECT task_id" in q else [])
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
    run = live.Run(mock=False, max_cost=1.0, build=False, keys=live.Keys({"OPENAI_API_KEY": KEY}))
    live.flow(run, {})
    assert clock.now - start > 40 * 60  # the run outlived four token lifetimes
    assert gateway.rejected == []  # no request carried an expired token
    assert run.deleted and gateway.deleted  # the deletion through the API still worked at the end
    assert run.report["status"] == "PASSED" and run.report["graded_outcome"] == "correct"
    assert gateway.subjects[gateway.lid] > 100  # every poll signed its own token


def test_cleanup_deletes_the_learner_with_a_fresh_token_long_after_the_lesson_began(long_lesson) -> None:
    clock, gateway = long_lesson
    run = live.Run(mock=False, max_cost=1.0, build=False, keys=live.Keys({"OPENAI_API_KEY": KEY}))
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
