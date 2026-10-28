"""The visual planner never sees the lesson's answer keys, so no image is searched for or generated from one
(issue #26).

Distinctive answers are planted in the lesson the teacher writes: the exercise and check-question answers, an exercise's
explanation and prompt, and a section example, explanation, narration, introduction and summary that repeat an
answer. None of them may reach the visual model's input, an image search query, an image generation prompt, the
stored VISUAL_PLAN or IMAGE_ASSET metadata. A model that guesses an answer anyway is sent back without the answer
quoted, and its guess never reaches a provider."""

from __future__ import annotations

import json

import pytest

from app.agents.base import OutputRejected
from app.agents.visual.agent import VisualAgent
from app.agents.visual.briefing import build_brief
from app.config.settings import Settings
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.image_search.mock import MockImageSearchProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders, make_teacher, visuals
from app.schemas.artifact import ArtifactType
from app.schemas.lesson import VisualPlanningInput, VisualRequest
from app.schemas.task import TaskStatus
from app.schemas.visual import (
    ANSWER_ISOLATED_PLANNING,
    ImageAsset,
    VisualPlan,
    VisualPlanProposal,
    VisualRequirement,
    VisualResult,
)
from app.services.container import build_container
from tests.auth import TestAuth, client_for
from tests.conftest import FIXTURES, LEARNER, run_lesson

CATALOG = FIXTURES / "image_catalog.json"
EXERCISE_ANSWER = "zebraquartz"
CHECK_ANSWER = "moonflake harbor"
ANSWERS = (EXERCISE_ANSWER, CHECK_ANSWER)
# Exercise text that is not an answer key but must not reach the planner either: exercises are left out entirely.
EXERCISE_ONLY = ("plumbridge", "glintwhistle")
PLANTED = (*ANSWERS, *EXERCISE_ONLY)
KEPT_EXAMPLE = "Me encanta el fútbol."


def planting_teacher(request) -> dict:
    """The mock teacher's lesson, with answers planted in it and repeated in learner-facing text."""
    lesson = make_teacher(True)(request)
    for i, ex in enumerate(lesson["exercises"]):
        ex["answer"] = EXERCISE_ANSWER if i == 0 else ex["answer"]
        ex["explanation"] = f"Because {EXERCISE_ONLY[0]} agrees with the subject."
        ex["prompt"] = f"{ex['prompt']} ({EXERCISE_ONLY[1]})"
    for q in lesson["check_questions"]:
        q["answer"] = CHECK_ANSWER
    first, second = lesson["sections"][0], lesson["sections"][1]
    first["examples"] = [f"Say {EXERCISE_ANSWER} out loud.", *first["examples"]]
    first["narration"] = f"{first['narration']} The answer is {EXERCISE_ANSWER}."
    second["explanation"] = f"{second['explanation']} Remember {CHECK_ANSWER.upper()}!"
    lesson["introduction"] = f"{lesson['introduction']} Hint: {CHECK_ANSWER}."
    lesson["summary"] = f"{lesson['summary']} Answer: {EXERCISE_ANSWER}."
    return lesson


class SpySearch(MockImageSearchProvider):
    def __init__(self) -> None:
        super().__init__(CATALOG)
        self.queries: list[str] = []

    async def search(self, request):
        self.queries.append(request.query)
        return await super().search(request)


class SpyGeneration(MockImageGenerationProvider):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    async def generate(self, request):
        self.prompts.append(" ".join(filter(None, [request.prompt, request.negative_prompt, request.style])))
        return await super().generate(request)


@pytest.fixture
def world(tmp_path):
    made = []

    def make(visual=visuals):
        llm = MockLLMProvider({**default_responders(), "teacher": planting_teacher, "visual": visual})
        search, generation = SpySearch(), SpyGeneration()
        c = build_container(Settings(data_dir=tmp_path / f"d{len(made)}", log_json=False),
                            llm_providers={"mock": llm}, image_search_provider=search,
                            image_generation_provider=generation)
        made.append(c)
        return c, llm, search, generation

    yield make
    for c in made:
        c.close()


def assert_absent(text: str, where: str, planted=PLANTED) -> None:
    lowered = text.lower()
    for secret in planted:
        assert secret.lower() not in lowered, f"{secret!r} reached {where}"


def visuals_of(task) -> VisualResult:
    return VisualResult.model_validate(task.workflow.node_states["visual_policy"].output)


def visual_requests(llm):
    return [r for r in llm.requests if r.agent_id == "visual"]


async def test_planted_answers_reach_no_planner_input_query_prompt_plan_or_asset(world) -> None:
    container, llm, search, generation = world()
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors

    # The answers were really planted in the stored lesson, which the agent's code holds.
    lesson = container.artifacts.read(container.artifacts.find(task.task_id, "lesson").artifact_id).decode()
    assert all(secret in lesson.lower() for secret in PLANTED)

    # The visual model's whole request (system prompt, rendered input, structured payload) holds none of them.
    for request in visual_requests(llm):
        assert_absent(json.dumps(request.input_payload, ensure_ascii=False), "the planner payload")
        assert_absent(request.system + "".join(m.content for m in request.messages), "the planner prompt")
    planning = VisualPlanningInput.model_validate(visual_requests(llm)[0].input_payload)
    assert [s.ref for s in planning.sections] == ["s1", "s2"]
    first, second = planning.sections
    assert first.examples == [e for e in first.examples if EXERCISE_ANSWER not in e] and first.examples
    assert second.explanation is None and second.heading  # the leaking explanation is dropped, the rest kept

    # Both image paths ran, and no provider ever received an answer.
    assert search.queries and generation.prompts
    for text in [*search.queries, *generation.prompts]:
        assert_absent(text, "an image provider")

    # Nor did the stored plan or the image metadata.
    plan_art = container.artifacts.find(task.task_id, "visual_plan")
    assert_absent(container.artifacts.read(plan_art.artifact_id).decode(), "the VISUAL_PLAN")
    images = [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.IMAGE_ASSET]
    assert {ImageAsset.model_validate(a.metadata).origin for a in images} == {"search", "generated"}
    for art in images:
        assert_absent(json.dumps(art.metadata, ensure_ascii=False), "IMAGE_ASSET metadata")
        assert art.metadata["planning"] == ANSWER_ISOLATED_PLANNING


def guessing(*, times: int):
    """A visual model that states an answer it inferred, in the search query and the prompt, `times` times."""
    calls = {"n": 0}

    def respond(request) -> dict:
        calls["n"] += 1
        proposal = visuals(request)
        if calls["n"] <= times:
            for r in proposal["requirements"]:
                if r.get("search_query"):
                    r["search_query"] = f"{r['search_query']} {EXERCISE_ANSWER}"
                if r.get("generation_prompt"):
                    r["generation_prompt"] = f"{r['generation_prompt']}, labelled {CHECK_ANSWER}"
        return proposal

    return respond


async def test_a_guessed_answer_is_sent_back_without_quoting_it_and_never_reaches_a_provider(world) -> None:
    container, llm, search, generation = world(guessing(times=1))
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    requests = visual_requests(llm)
    assert len(requests) == 2  # rejected once, then corrected
    # The model's own rejected output is replayed to it; the system's correction and the input never add an answer.
    assert [m.role for m in requests[1].messages] == ["user", "assistant", "user"]
    prompt, _own_output, correction = (m.content for m in requests[1].messages)
    assert "state the answer" in correction and "search_query" in correction
    assert_absent(requests[1].system + prompt + correction, "the correction sent to the model")
    assert search.queries and generation.prompts
    for text in [*search.queries, *generation.prompts]:
        assert_absent(text, "an image provider")


async def test_a_model_that_keeps_guessing_produces_no_image_and_no_provider_call(world) -> None:
    container, llm, search, generation = world(guessing(times=99))
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED
    assert search.queries == [] and generation.prompts == []
    assert not [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.IMAGE_ASSET]


# --- the brief itself ------------------------------------------------------------------------------------------


@pytest.fixture
async def visual_request(world) -> VisualRequest:
    """The VisualRequest of a planted lesson, as the workflow built it."""
    container, llm, _, _ = world()
    task = await run_lesson(container)
    read = container.artifacts.read
    find = lambda name: read(container.artifacts.find(task.task_id, name).artifact_id).decode()  # noqa: E731
    return VisualRequest.model_validate({"lesson": json.loads(find("lesson")), "plan": json.loads(find("lesson_plan")),
                                         "research": json.loads(find("research_bundle"))})


async def test_brief_holds_only_teaching_text_without_answers(visual_request) -> None:
    brief = build_brief(visual_request)
    dumped = brief.planning.model_dump_json()
    assert_absent(dumped, "the brief")
    assert set(VisualPlanningInput.model_fields) == {
        "lesson_title", "subject", "topic", "level", "objectives", "sections", "max_visuals",
        "max_generated_images", "max_searched_images"}
    lesson = visual_request.lesson
    for forbidden in [lesson.introduction, lesson.summary, *(s.section_id for s in lesson.sections),
                      *(s.narration for s in lesson.sections)]:
        assert forbidden not in dumped
    assert brief.sections == {"s1": lesson.sections[0].section_id, "s2": lesson.sections[1].section_id}
    assert set(brief.answers) >= set(ANSWERS)
    assert KEPT_EXAMPLE in dumped


async def test_practice_and_assessment_sections_and_answer_named_concepts_are_left_out(visual_request) -> None:
    lesson = visual_request.lesson
    practice = lesson.sections[0].model_copy(update={"purpose": "guided_practice"})
    named = lesson.sections[1]  # its heading is fine, but its concept's name states the answer
    plan = visual_request.plan.model_copy(update={"concepts": [
        c.model_copy(update={"name": f"The {EXERCISE_ANSWER} rule"}) if c.concept_id == named.concept_id else c
        for c in visual_request.plan.concepts]})
    brief = build_brief(visual_request.model_copy(update={
        "plan": plan, "lesson": lesson.model_copy(update={"sections": [practice, named]})}))
    assert brief.planning.sections == [] and brief.sections == {}


async def test_check_rejects_every_answer_bearing_field_without_quoting_the_answer(visual_request) -> None:
    brief = build_brief(visual_request)
    clean = VisualPlanProposal.model_validate(visuals(type("R", (), {"input_payload": brief.planning.model_dump(
        mode="json")})()))
    VisualAgent.check(None, clean, brief)
    for field in ("purpose", "concept", "description", "search_query", "generation_prompt"):
        reqs = [r.model_copy() for r in clean.requirements]
        target = next(i for i, r in enumerate(reqs) if getattr(r, field))
        reqs[target] = reqs[target].model_copy(update={field: f"{getattr(reqs[target], field)} {CHECK_ANSWER.title()}"})
        with pytest.raises(OutputRejected) as exc:
            VisualAgent.check(None, clean.model_copy(update={"requirements": reqs}), brief)
        assert f"requirement {target + 1}.{field}" in str(exc.value)  # by position, never by a model-written value
        assert_absent(str(exc.value), "the rejection message")
    section_id = visual_request.lesson.sections[0].section_id
    with pytest.raises(OutputRejected, match=r"requirements \[1\] reference unknown lesson sections") as exc:
        VisualAgent.check(None, clean.model_copy(update={"requirements": [
            clean.requirements[0].model_copy(update={"lesson_section_id": section_id})]}), brief)
    assert section_id not in str(exc.value)


# --- B1: model-written visual ids ------------------------------------------------------------------------------


def naming(answer: str, *, leak_query_times: int = 0):
    """A visual model that names every visual after the answer (and, `leak_query_times` times, also puts it in the
    search query)."""
    calls = {"n": 0}

    def respond(request) -> dict:
        calls["n"] += 1
        proposal = visuals(request)
        for i, r in enumerate(proposal["requirements"]):
            r["visual_id"] = f"{answer}.{i}"
            if calls["n"] <= leak_query_times and r.get("search_query"):
                r["search_query"] = f"{r['search_query']} {answer}"
        return proposal

    return respond


def learner_visual_listing(container, task) -> str:
    """The learner's artifact listing, for the visual stage's artifacts (the plan and its images)."""
    with client_for(container, LEARNER["learner_id"], auth=TestAuth()) as a:
        r = a.get(f"/tasks/{task.task_id}/artifacts")
    assert r.status_code == 200
    visual = [v for v in r.json() if v["type"] in (ArtifactType.IMAGE_ASSET.value, ArtifactType.VISUAL_PLAN.value)]
    assert len(visual) == 5
    return json.dumps(visual, ensure_ascii=False)


async def test_a_visual_id_the_model_writes_is_ignored_and_never_stored_or_quoted(world) -> None:
    container, llm, search, generation = world(naming(EXERCISE_ANSWER))
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    assert len(visual_requests(llm)) == 1  # accepted as is: the id was dropped unread, nothing to correct

    plan = VisualPlan.model_validate_json(container.artifacts.read(
        container.artifacts.find(task.task_id, "visual_plan").artifact_id))
    assert [r.visual_id for r in plan.requirements] == ["v1_photo", "v1_diagram", "v2_illustration", "v2_diagram"]
    assert_absent(plan.model_dump_json(), "the VISUAL_PLAN")
    images = [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.IMAGE_ASSET]
    assert len(images) == 4
    for art in images:
        assert_absent(art.name, "an artifact name")
        assert_absent(json.dumps(art.metadata, ensure_ascii=False), "IMAGE_ASSET metadata")
        assert art.metadata["planning"] == ANSWER_ISOLATED_PLANNING  # what was stored is clean, so it is marked
    assert_absent(learner_visual_listing(container, task), "the learner's artifact listing")
    assert_absent(json.dumps(visuals_of(task).model_dump(mode="json")), "the visual result")
    for text in [*search.queries, *generation.prompts]:
        assert_absent(text, "an image provider")


async def test_a_correction_names_rejected_items_by_position_even_when_the_model_named_them(world) -> None:
    container, llm, _, _ = world(naming(EXERCISE_ANSWER, leak_query_times=1))
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    requests = visual_requests(llm)
    assert len(requests) == 2
    correction = requests[1].messages[-1].content
    assert "requirement 1.search_query" in correction
    assert_absent(requests[1].system + requests[1].messages[0].content + correction, "the correction")
    for art in container.task_service.artifacts(task.task_id):
        assert_absent(art.name, "an artifact name")


async def test_images_of_a_plan_that_states_an_answer_get_no_isolation_marker(world, monkeypatch) -> None:
    """Defence in depth: should a leaking plan ever get past `check`, its images are not marked and not served."""
    monkeypatch.setattr(VisualAgent, "check", lambda self, output, source: None)

    def leaky(request) -> dict:
        proposal = visuals(request)
        proposal["requirements"][0]["description"] += f" Shows {EXERCISE_ANSWER}."
        return proposal

    container, _, _, _ = world(leaky)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    images = [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.IMAGE_ASSET]
    assert images and all(a.metadata.get("planning") is None for a in images)
    with client_for(container, LEARNER["learner_id"], auth=TestAuth()) as a:
        for image in images:
            assert a.get(f"/artifacts/{image.artifact_id}/content").status_code == 404


async def test_isolation_holds_only_when_every_model_derived_field_is_clean(visual_request) -> None:
    brief = build_brief(visual_request)
    proposal = VisualPlanProposal.model_validate(visuals(type("R", (), {"input_payload": brief.planning.model_dump(
        mode="json")})()))
    req = VisualRequirement(**proposal.requirements[0].model_dump(exclude={"lesson_section_id"}), visual_id="v1_photo",
                            lesson_section_id=visual_request.lesson.sections[0].section_id)
    plan = VisualPlan(plan_id="vp_1", lesson_title="", requirements=[req], rationale="r")
    assert brief.isolated(plan)
    for leak in ({"visual_id": f"{EXERCISE_ANSWER}.1"}, {"description": CHECK_ANSWER}, {"aspect_ratio": "16:9"}):
        changed = plan.model_copy(update={"requirements": [req.model_copy(update=leak)]})
        assert brief.isolated(changed) is ("aspect_ratio" in leak)
    assert not brief.isolated(plan.model_copy(update={"lesson_title": f"All about {EXERCISE_ANSWER}"}))
    assert not brief.isolated(plan.model_copy(update={"rationale": CHECK_ANSWER}))
