from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.learner import LearnerProfileInput
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer
from app.schemas.task import Task, TaskStatus
from app.services.container import Container, build_container

# Lesson-flow tests run the whole workflow, video included; they use the mock composer so the suite does not encode
# a full-HD MP4 per test. The real FFmpeg composer is exercised by tests/integration/test_video_ffmpeg.py and the
# video demo, which select it explicitly.
os.environ.setdefault("TA_VIDEO_COMPOSER", "mock")
# Tests run offline on mock providers: no network provider may run and no credentials are needed. The optional real
# provider smoke tests (RUN_PROVIDER_SMOKE_TESTS=true) and live end-to-end test (RUN_LIVE_E2E=true) are the exceptions.
# Provider choices from the developer's environment (or .env) are overridden, so the suite always runs on mocks.
LIVE = any(os.environ.get(flag, "").lower() == "true" for flag in ("RUN_PROVIDER_SMOKE_TESTS", "RUN_LIVE_E2E"))
if not LIVE:
    os.environ["TEACHING_AGENT_OFFLINE"] = "true"
    for _key in [k for k in os.environ if re.match(r"^LLM_[A-Z0-9_]+_(PROVIDER|MODEL)$", k)]:
        del os.environ[_key]
    os.environ.update({"LLM_PROVIDER": "", "LLM_MODEL": "", "TTS_PROVIDER": "mock", "TTS_FALLBACK_PROVIDER": "",
                       "IMAGE_PROVIDER": "mock", "IMAGE_FALLBACK_PROVIDER": "", "IMAGE_SEARCH_PROVIDER": "mock",
                       "SEARCH_PROVIDER": "mock", "SEARCH_FALLBACK_PROVIDER": ""})

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


EVALUATION_ANSWERS = json.loads((FIXTURES / "evaluation_answers.json").read_text(encoding="utf-8"))


def evaluation_answers_for(task: Task, key: dict | None = None) -> dict:
    """The fixture learner's answers to a WAITING evaluation task, as the POST /tasks/{id}/answers body."""
    assert task.waiting is not None and task.waiting.kind == "assessment_answers"
    by_concept = (key or EVALUATION_ANSWERS)["answers"]
    return {"answers": [{"question_id": q["question_id"], "answer": by_concept[q["concept_id"]][q["kind"]]}
                        for q in task.waiting.prompt["questions"]]}


async def start_evaluation(container: Container) -> tuple[Task, Task]:
    """Complete a lesson, then start its evaluation. Returns (lesson task, evaluation task WAITING for answers)."""
    lesson = await run_lesson(container)
    assert lesson.status == TaskStatus.COMPLETED, lesson.errors
    evaluation = await container.task_service.start_evaluation(lesson.task_id, user_id="u1")
    return lesson, evaluation
