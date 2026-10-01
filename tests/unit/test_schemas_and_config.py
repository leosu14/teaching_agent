from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config.routing import ConfigError, load_routing
from app.config.settings import REPO_ROOT, Settings
from app.runtime.tasks.state_machine import InvalidTransition, transition
from app.schemas.lesson import (
    DiagnosticStep,
    LessonPlan,
    ReviewResult,
)
from app.schemas.presentation import BulletBlock
from app.schemas.task import Task, TaskStatus


def test_lesson_plan_minutes_must_add_up() -> None:
    base = dict(title="t", level="A2", objectives=["o"], teaching_strategy="s",
                concepts=[{"concept_id": "c", "name": "c", "rationale": "r", "strategy": "s"}],
                sequence=[{"step_id": "1", "concept_id": "c", "activity": "explain", "minutes": 5}])
    assert LessonPlan(**base, estimated_minutes=5)
    with pytest.raises(ValidationError, match="sum of sequence"):
        LessonPlan(**base, estimated_minutes=6)
    with pytest.raises(ValidationError, match="unknown concept"):
        LessonPlan(**{**base, "sequence": [{"step_id": "1", "concept_id": "zz", "activity": "explain", "minutes": 5}]},
                   estimated_minutes=5)


def test_review_verdict_must_match_issues() -> None:
    issue = {"issue_id": "i", "criterion": "structure", "severity": "major", "location": "x", "problem": "p",
             "suggested_fix": "f"}
    with pytest.raises(ValidationError):
        ReviewResult(verdict="APPROVED", scores={}, issues=[issue], summary="")
    with pytest.raises(ValidationError):
        ReviewResult(verdict="REVISION_REQUIRED", scores={}, issues=[], summary="")
    with pytest.raises(ValidationError):
        ReviewResult(verdict="APPROVED", scores={"structure": 1.5}, summary="")


def test_slides_limit_density_and_diagnostic_steps_are_consistent() -> None:
    with pytest.raises(ValidationError):
        BulletBlock(items=["b"] * 7)
    with pytest.raises(ValidationError, match="20 words"):
        BulletBlock(items=[" ".join(["w"] * 21)])
    with pytest.raises(ValidationError, match="requires"):
        DiagnosticStep(status="ask", concepts=[])
    with pytest.raises(ValidationError):
        LessonPlan.model_validate_json('{"title": "t", "unexpected": 1}')


def test_task_state_machine() -> None:
    task = Task(task_id="t", user_id="u", learner_id="l", request="r")
    for status in (TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.REVIEWING, TaskStatus.RUNNING,
                   TaskStatus.WAITING, TaskStatus.RUNNING, TaskStatus.COMPLETED):
        transition(task, status)
    with pytest.raises(InvalidTransition):
        transition(task, TaskStatus.RUNNING)
    fresh = Task(task_id="t", user_id="u", learner_id="l", request="r")
    with pytest.raises(InvalidTransition):
        transition(fresh, TaskStatus.COMPLETED)


def test_configuration_is_validated(tmp_path, monkeypatch) -> None:
    routing = load_routing(REPO_ROOT / "config" / "routing.toml")
    assert all(routing.pricing[t.model].output_per_mtok > 0 for ts in routing.tiers.values() for t in ts)
    with pytest.raises(ConfigError, match="not found"):
        load_routing(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[tiers]\nreasoning = 3\n")
    with pytest.raises(ConfigError, match="invalid"):
        load_routing(bad)
    with pytest.raises(ConfigError, match="missing"):
        Settings(corpus_dir=tmp_path).validate_runtime()
    monkeypatch.setenv("TA_MAX_REVISIONS", "4")
    monkeypatch.setenv("TA_LLM_PROVIDERS", '["mock"]')
    assert Settings().max_revisions == 4
