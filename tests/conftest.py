from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.learner import LearnerProfileInput
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer
from app.schemas.task import Task, TaskStatus
from app.services.container import Container, build_container

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "fixtures" / "demo"
LEARNER = json.loads((FIXTURES / "learner.json").read_text(encoding="utf-8"))
ANSWERS = json.loads((FIXTURES / "answers.json").read_text(encoding="utf-8"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", log_json=False)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider(default_responders())


@pytest.fixture
def container(settings: Settings, mock_llm: MockLLMProvider):
    c = build_container(settings, llm_providers={"mock": mock_llm})
    yield c
    c.close()


def add_demo_learner(container: Container) -> str:
    container.learner_service.upsert(LEARNER["learner_id"], LearnerProfileInput.model_validate(LEARNER["profile"]))
    return LEARNER["learner_id"]


def answers_for(task: Task) -> DiagnosticAnswers:
    assert task.waiting is not None
    sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
    key = ANSWERS["rounds"][sheet.round_number - 1]
    return DiagnosticAnswers(answers=[LearnerAnswer(question_id=q.question_id, answer=key.get(q.concept_id, ""))
                                      for q in sheet.questions])


async def run_lesson(container: Container, request: str = LEARNER["request"]) -> Task:
    learner_id = add_demo_learner(container)
    task = await container.task_service.create_and_run(request=request, learner_id=learner_id, user_id="u1")
    while task.status == TaskStatus.WAITING:
        task = await container.task_service.submit_assessment(task.task_id, answers_for(task))
    return task
