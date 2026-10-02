"""`python scripts/run_production_demo.py`: dry run, refusal, confirmation, a full run on mocks (the same path a
production run takes), reuse of an identical run, and evaluation. No credentials and no network."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from app.config.settings import Settings
from app.services.container import build_container
from tests.conftest import REPO_ROOT

SCRIPT = "scripts/run_production_demo.py"
SECRET = "sk-live-should-never-print-0123456789"


def run(*args: str, env: dict | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    environ = {k: v for k, v in os.environ.items() if k != "TEACHING_AGENT_OFFLINE"}
    environ.update(env or {})
    return subprocess.run([sys.executable, SCRIPT, *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=timeout, env=environ)


def test_dry_run_offline_checks_everything_and_calls_nothing(tmp_path) -> None:
    proc = run("--dry-run", "--data-dir", str(tmp_path / "data"), "--output", str(tmp_path / "out"),
               env={"OPENAI_API_KEY": SECRET})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("Level:     B1 (cefr)", "Topic:     Climate change", "Mode:      offline",
                     "Knowledge base: 3 concepts", "compose_video", "set TEACHING_AGENT_MODE=production",
                     "no provider was called", "a production run is NOT possible"):
        assert expected in out, expected
    assert SECRET not in proc.stdout + proc.stderr
    assert not (tmp_path / "out").exists()  # a dry run writes nothing


def test_production_run_is_refused_when_not_configured(tmp_path) -> None:
    proc = run("--yes", "--data-dir", str(tmp_path / "data"), "--output", str(tmp_path / "out"),
               env={"TEACHING_AGENT_MODE": "production", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-test",
                    "OPENAI_API_KEY": SECRET, "TTS_PROVIDER": "openai", "SEARCH_PROVIDER": "", "IMAGE_PROVIDER": ""})
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "SEARCH_PROVIDER" in proc.stdout and "IMAGE_PROVIDER" in proc.stdout
    assert "Production run refused" in proc.stdout and SECRET not in proc.stdout + proc.stderr


READY = {"TEACHING_AGENT_MODE": "production", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-test",
         "OPENAI_API_KEY": SECRET, "TTS_PROVIDER": "openai", "IMAGE_PROVIDER": "openai", "SEARCH_PROVIDER": "tavily",
         "TAVILY_API_KEY": "tvly-should-never-print-0123", "OPENAI_BASE_URL": "http://127.0.0.1:9",
         "SEARCH_BASE_URL": "http://127.0.0.1:9"}  # unreachable: nothing may be sent


def test_without_confirmation_a_ready_production_run_calls_nothing(tmp_path) -> None:
    proc = run("--data-dir", str(tmp_path / "data"), "--output", str(tmp_path / "out"), env=READY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Mode:      production" in proc.stdout and "PROBLEM" not in proc.stdout
    assert "No provider was called" in proc.stdout and "--confirm" in proc.stdout
    assert not (tmp_path / "out").exists() and SECRET not in proc.stdout + proc.stderr


def test_confirmed_run_writes_the_report_and_video_and_an_identical_run_is_reused(tmp_path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    answers = tmp_path / "diagnostic.json"
    answers.write_text(json.dumps({"es.climate.vocabulary": "el calentamiento global"}), encoding="utf-8")
    log = tmp_path / "production.log"
    proc = run("--mock", "--confirm", "--skip-evaluation", "--export", "--diagnostic-answers", str(answers),
               "--log-file", str(log), "--data-dir", str(data), "--output", str(out), env={"OPENAI_API_KEY": SECRET})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for expected in ("Status    COMPLETED", "Graph:     complete", "Usage:", "production_run.json"):
        assert expected in proc.stdout, expected

    report = json.loads((out / "production_run.json").read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETED" and report["graph_check"]["complete"]
    assert (report["level"], report["topic"], report["language"]) == ("B1", "Climate change", "es")
    for key in ("task_id", "learner_id", "start_time", "end_time", "providers", "artifact_graph", "usage",
                "estimated_cost_usd", "warnings", "errors", "final_video", "trace", "budget"):
        assert key in report, key
    video = report["final_video"]
    assert video["duration"] > 10 and (video["width"], video["height"]) == (1920, 1080)
    assert sorted(p.name for p in out.iterdir()) == ["lesson.mp4", "lesson.pptx", "lesson.vtt",
                                                     "production_run.json"]  # only what was asked for
    assert (out / "lesson.mp4").stat().st_size == video["size_bytes"]
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(r.get("request_id", "").startswith("preq_") and r["status"] == "ok" for r in lines)
    everything = proc.stdout + proc.stderr + log.read_text(encoding="utf-8") + json.dumps(report)
    assert SECRET not in everything and "Authorization" not in everything

    again = run("--mock", "--yes", "--skip-evaluation", "--data-dir", str(data), "--output", str(out))
    assert again.returncode == 0, again.stdout + again.stderr
    second = json.loads((out / "production_run.json").read_text(encoding="utf-8"))
    assert second["reused"] and second["task_id"] == report["task_id"]
    assert second["final_video"]["content_hash"] == video["content_hash"]


def test_evaluation_mode_grades_the_learner(tmp_path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    env = {"TA_VIDEO_COMPOSER": "mock"}
    proc = run("--mock", "--yes", "--data-dir", str(data), "--output", str(out), env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((out / "production_run.json").read_text(encoding="utf-8"))
    assert report["evaluation_task_id"] and "--evaluate" in proc.stdout

    c = build_container(Settings(data_dir=data, log_json=False))
    try:
        questions = c.task_service.get(report["evaluation_task_id"]).waiting.prompt["questions"]
    finally:
        c.close()
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({q["question_id"]: (q.get("choices") or ["no sé"])[0] for q in questions}),
                       encoding="utf-8")
    graded = run("--mock", "--evaluate", report["task_id"], "--answers", str(answers), "--data-dir", str(data),
                 env=env)
    assert graded.returncode == 0, graded.stdout + graded.stderr
    assert "COMPLETED" in graded.stdout and "score" in graded.stdout
