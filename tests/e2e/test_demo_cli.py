"""`python scripts/run_demo.py` is the one command that proves the slice works."""

from __future__ import annotations

import subprocess
import sys

from tests.conftest import REPO_ROOT


def test_demo_script_runs_to_completion(tmp_path) -> None:
    proc = subprocess.run([sys.executable, "scripts/run_demo.py", "--data-dir", str(tmp_path)], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("task_id:", "final status:  COMPLETED", "workflow steps:", "generated artifacts:",
                     "learner mastery changes:", "token usage:", "estimated cost:", "actual cost:",
                     "WAITING for diagnostic round 1", "WAITING for diagnostic round 2",
                     "review:        APPROVED after 1 revision(s)"):
        assert expected in out, expected
    assert (tmp_path / "teaching_agent.db").exists()
    assert any((tmp_path / "objects").rglob("v1.json"))


def test_evaluation_demo_script_runs_to_completion(tmp_path) -> None:
    proc = subprocess.run([sys.executable, "scripts/run_evaluation_demo.py", "--data-dir", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("final status:  COMPLETED", "lesson completed:", "assessment generated:",
                     "task is WAITING for assessment_answers", "answers submitted; task status: COMPLETED",
                     "evaluation result:", "mastery before -> after:", "remaining gaps:  es.football.opinions",
                     "next recommendation: reteach"):
        assert expected in out, expected
    assert any((tmp_path / "objects").rglob("learner_evaluation/v1.json"))


def test_research_demo_script_runs_to_completion(tmp_path) -> None:
    proc = subprocess.run([sys.executable, "scripts/run_research_demo.py", "--data-dir", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for expected in ("1. Research request", "2. Generated queries", "3. Mock search results", "4. Selected sources",
                     "5. Extracted evidence", "6. Citations", "7. ResearchBundle", "8. Lesson generated from the bundle",
                     "status:       complete", "reliability 0.30 is below 0.50", "duplicate: https://www.sports-news",
                     "[c6] -> ev6 ->", "search:mock"):
        assert expected in out, expected
    assert any((tmp_path / "objects").rglob("research_bundle/v1.json"))
