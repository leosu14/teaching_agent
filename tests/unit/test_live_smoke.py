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
            "TA_VENDOR_CAPS": "openai=50,tavily=10"}


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
            "ceiling_usd": 1.0, "by_capability": {c: {"cost_usd": 0.1} for c in ("llm", "tts", "image", "search")}}
    live.assert_ledger(good, mock=True)
    for change in ({"total_usd": 1.01}, {"entries_still_reserved": 1}, {"attempts_outside_the_cap": 2},
                   {"unpriced_attempts": 1}, {"total_usd": 0.0}, {"by_capability": {"llm": {"cost_usd": 1}}}):
        with pytest.raises(AssertionError):
            live.assert_ledger({**good, **change}, mock=True)


def test_the_compose_command_adds_the_live_override_under_its_own_project() -> None:
    cmd = live.compose_command()
    assert cmd[-2:] == ["-p", "ta-live"] and str(Path("deploy") / "compose.live.yaml") in " ".join(cmd)
