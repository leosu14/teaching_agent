"""The bundled Spanish A1 to B1 course end to end (MVP item 5), on mock providers: TA_COURSES makes the course the
knowledge base; goals up to a level target the course's concepts; a curriculum is planned over its prerequisite
graph; the next action's lesson runs with the course's diagnostic probes and research material; every unit's topic
has its concept map and passes the production preflight's knowledge check; a topic outside the course is refused."""

from __future__ import annotations

import json

import pytest

from app.config.settings import COURSES_DIR, Settings
from app.content.course import read_course
from app.learner.frameworks import CEFR
from app.observability.scope import ExecutionScope, UsageLedger
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.curriculum import GoalInput, LearningActionType
from app.schemas.learner import LearnerProfileInput
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer
from app.schemas.production import ProductionTask
from app.schemas.task import Task, TaskStatus
from app.services.container import build_container
from app.tools.base import ToolCaller
from tests.conftest import LEARNER as DEMO_LEARNER

COURSE = read_course(COURSES_DIR / "spanish-a1-b1.json")
UNITS = {u.topic: u for u in COURSE.units}
KEYS = {p.prompt: p.answer for _, c in COURSE.concepts() for p in c.probes}
LEARNER = "course-learner"


@pytest.fixture
def course_container(tmp_path):
    # The mock image search reads the demo picture catalog (football and climate scenes), which has no picture for
    # most course concepts; a missing searched picture must not fail a mock lesson. Production plans generated
    # visuals only (MAX_SEARCHED_IMAGES=0).
    settings = Settings(data_dir=tmp_path / "data", log_json=False, courses="spanish-a1-b1",
                        visual_failure_policy="continue")
    c = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    c.learner_service.upsert(LEARNER, LearnerProfileInput.model_validate(DEMO_LEARNER["profile"]))
    yield c
    c.close()


async def answer_all(container, task: Task, *, correct: bool = True) -> Task:
    """Answer each diagnostic round from the course's probe keys (or wrongly)."""
    while task.status == TaskStatus.WAITING and task.waiting.kind == "diagnostic_answers":
        sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
        assert all(q.prompt in KEYS for q in sheet.questions), "the diagnostic asks the course's probes"
        answers = [LearnerAnswer(question_id=q.question_id, answer=KEYS[q.prompt] if correct else "no sé")
                   for q in sheet.questions]
        task = await container.task_service.submit_assessment(task.task_id, DiagnosticAnswers(answers=answers))
    return task


def ancestors(ids: set[str]) -> set[str]:
    prereqs = {c.concept_id: c.prerequisites for _, c in COURSE.concepts()}
    out: set[str] = set()
    stack = [p for i in ids for p in prereqs[i]]
    while stack:
        cid = stack.pop()
        if cid not in out:
            out.add(cid)
            stack += prereqs[cid]
    return out


def up_to(level: str) -> set[str]:
    top = CEFR.index(level)
    return {c.concept_id for u, c in COURSE.concepts() if CEFR.index(COURSE.level_of(u, c)) <= top}


def scope(container) -> ExecutionScope:
    return ExecutionScope(events=container.events, usage=UsageLedger())


def content(container, artifact) -> dict:
    return json.loads(container.artifacts.read(artifact.artifact_id))


async def test_the_course_replaces_the_fixture_knowledge_base(course_container) -> None:
    caller = ToolCaller(caller_id="test", allowed_tools=frozenset({"knowledge.concepts"}),
                        permissions=frozenset({"knowledge:read"}))
    found = await course_container.tools.call(caller, "knowledge.concepts", {"domain": "spanish"},
                                              scope(course_container))
    ids = [c.concept_id for c in found.concepts]
    assert set(ids) == {c.concept_id for _, c in COURSE.concepts()} and len(ids) == len(set(ids))
    assert not any(cid.startswith("es.football") for cid in ids)


@pytest.mark.parametrize("level", ["A1", "A2", "B1"])
async def test_a_level_goal_targets_the_course_concepts_up_to_that_level(course_container, level) -> None:
    goal, created = await course_container.curriculum_service.create_goal(
        LEARNER, GoalInput(title=f"Reach {level}", domain="spanish", target_level=level))
    assert created and set(goal.target_concepts) == up_to(level) and goal.target_source == "level"


async def test_goal_curriculum_next_action_and_lesson_run_on_the_course(course_container) -> None:
    service = course_container.curriculum_service
    goal, _ = await service.create_goal(LEARNER, GoalInput(title="Reach A2", domain="spanish", target_level="A2"))
    build = await service.build_curriculum(goal.goal_id, user_id="u1")
    assert build.status == TaskStatus.COMPLETED, build.errors
    curriculum = service.curriculum(goal.goal_id)
    order = [o.concept_id for o in curriculum.objectives]
    assert set(order) == up_to("A2")
    for o in curriculum.objectives:  # prerequisites first, as the course defines them
        concept = next(c for _, c in COURSE.concepts() if c.concept_id == o.concept_id)
        assert all(order.index(p) < order.index(o.concept_id) for p in concept.prerequisites)

    action = await service.next_action(LEARNER)
    assert action.action == LearningActionType.LEARN and action.concept_id == order[0]
    lesson = await answer_all(course_container, await service.start_lesson(action, user_id="u1"))
    assert lesson.status == TaskStatus.COMPLETED, lesson.errors
    arts = {a.name: a for a in course_container.task_service.artifacts(lesson.task_id)}
    assert content(course_container, arts["pedagogical_plan"])["target_concepts"] == [action.concept_id]
    research = json.dumps(content(course_container, arts["research_bundle"]))
    assert f"kb://spanish-a1-b1/{action.concept_id}/reference" in research  # the course's teaching material


@pytest.mark.parametrize("topic", ["daily routine", "wishes advice and opinions"])
async def test_a_requested_lesson_on_a_course_topic_completes(course_container, topic) -> None:
    unit = UNITS[topic]
    task = await course_container.task_service.create_and_run(
        request=f"Teach me {unit.level} Spanish about {topic}", learner_id=LEARNER, user_id="u1")
    task = await answer_all(course_container, task, correct=False)
    assert task.status == TaskStatus.COMPLETED, task.errors
    arts = {a.name: a for a in course_container.task_service.artifacts(task.task_id)}
    # The learner missed everything, so the plan may target a prerequisite first: the unit's concepts or theirs.
    allowed = {c.concept_id for c in unit.concepts} | ancestors({c.concept_id for c in unit.concepts})
    targets = set(content(course_container, arts["pedagogical_plan"])["target_concepts"])
    assert targets and targets <= allowed


async def test_every_unit_has_its_concept_map_and_passes_the_knowledge_preflight(course_container) -> None:
    caller = ToolCaller(caller_id="test", allowed_tools=frozenset({"rag.concept_map"}),
                        permissions=frozenset({"knowledge:read"}))
    for unit in COURSE.units:
        found = await course_container.tools.call(caller, "rag.concept_map", {
            "subject": "spanish", "topic": unit.topic, "level": unit.level}, scope(course_container))
        assert {e.concept.concept_id for e in found.concepts} == {c.concept_id for c in unit.concepts}, unit.topic
        assert all(len(e.probes) >= 2 for e in found.concepts)
        plan = await course_container.production.plan(
            ProductionTask(level=unit.level, topic=unit.topic, language="es", learner_id=LEARNER),
            require_production=False)
        assert plan.knowledge_concepts == len(unit.concepts), unit.topic
        assert not any("knowledge base" in p for p in plan.problems), plan.problems


async def test_a_topic_outside_the_course_is_refused_at_preflight_naming_the_courses(course_container) -> None:
    plan = await course_container.production.plan(
        ProductionTask(level="B1", topic="Climate change", language="es", learner_id=LEARNER),
        require_production=False)
    assert plan.knowledge_concepts == 0 and not plan.ready
    problem = next(p for p in plan.problems if "knowledge base" in p)
    assert "spanish-a1-b1.json (TA_COURSES)" in problem and "add a unit for the topic to a course" in problem


def test_the_course_file_is_what_the_application_loads() -> None:
    raw = json.loads((COURSES_DIR / "spanish-a1-b1.json").read_text(encoding="utf-8"))
    assert raw["course_id"] == COURSE.course_id and len(raw["units"]) == len(COURSE.units)
