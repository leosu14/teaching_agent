"""The Visual step inside the lesson workflow: planning, search/generation, validation, IMAGE_ASSET artifacts,
review gating, failure policy, events, cost, resume and the separation from the textual citation chain."""

from __future__ import annotations

import json

import pytest

from app.config.settings import Settings
from app.providers.image.base import ImageGenerationProviderError
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.image_search.base import ImageSearchPage, ImageSearchProviderError
from app.providers.image_search.mock import MockImageSearchProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.artifact import ArtifactType
from app.schemas.learner import LearnerProfileInput
from app.schemas.lesson import LessonContent, VisualPlanningInput
from app.schemas.research import ResearchBundle
from app.schemas.task import TaskStatus
from app.schemas.visual import ImageAsset, ImageUsage, VisualPlan, VisualResult
from app.services.container import build_container
from app.utils.images import probe_image
from tests.conftest import FIXTURES, LEARNER, add_demo_learner, answers_for, run_lesson
from tests.integration.test_resume import SimulatedCrash, crash_after
from tests.integration.test_review_loop import always_reject

CATALOG = FIXTURES / "image_catalog.json"
VISUAL_EVENTS = ["visual.started", "visual.plan_created", "image.search_started", "image.search_completed",
                 "image.selected", "image.generation_started", "image.generation_completed",
                 "image.validation_failed", "image.asset_created", "visual.completed"]


class SearchWithout(MockImageSearchProvider):
    """Returns nothing for queries containing `word` (or fails, when `fail`)."""

    def __init__(self, word: str = "", fail: bool = False) -> None:
        super().__init__(CATALOG)
        self.word, self.fail = word, fail

    async def search(self, request):
        if self.word.lower() in request.query.lower():
            if self.fail:
                raise ImageSearchProviderError("image search API unavailable", transient=False)
            return ImageSearchPage(hits=[], usage=ImageUsage(requests=1, results=0))
        return await super().search(request)


class GenerationFailing(MockImageGenerationProvider):
    """Fails for prompts starting with `prefix` ('' = always); `transient_once` fails only the first call."""

    def __init__(self, prefix: str = "", transient_once: bool = False) -> None:
        self.prefix, self.transient_once, self.calls = prefix, transient_once, 0

    async def generate(self, request):
        self.calls += 1
        if self.transient_once:
            if self.calls == 1:
                raise ImageGenerationProviderError("GPU busy", transient=True)
        elif request.prompt.startswith(self.prefix):
            raise ImageGenerationProviderError("image generation API unavailable", transient=False)
        return await super().generate(request)


@pytest.fixture
def make_container(tmp_path):
    made = []

    def make(*, llm: MockLLMProvider | None = None, observers=(), data_dir=None, **kwargs):
        providers = {k: kwargs.pop(k) for k in ("image_search_provider", "image_generation_provider") if k in kwargs}
        c = build_container(Settings(data_dir=data_dir or tmp_path / f"d{len(made)}", log_json=False, **kwargs),
                            llm_providers={"mock": llm or MockLLMProvider(default_responders())},
                            observers=observers, **providers)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def visuals_of(task) -> VisualResult:
    return VisualResult.model_validate(task.workflow.node_states["visual_policy"].output)


def artifacts_by_name(container, task) -> dict:
    return {a.name: a for a in container.task_service.artifacts(task.task_id)}


def lesson_of(container, task) -> LessonContent:
    return LessonContent.model_validate_json(
        container.artifacts.read(container.artifacts.find(task.task_id, "lesson").artifact_id))


async def test_end_to_end_lesson_with_visual_assets(make_container) -> None:
    llm = MockLLMProvider(default_responders())
    container = make_container(llm=llm)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    order = [n for n in task.workflow.execution_order
             if n in {"diagnostic", "research", "plan", "teach_review", "visual_gate", "visual", "visual_policy",
                      "slides", "store_artifacts"}]
    assert order == ["diagnostic", "research", "plan", "teach_review", "visual_gate", "visual", "visual_policy",
                     "slides", "store_artifacts"]

    # Visual planning got the lesson plan, the approved lesson and the research bundle.
    planning = VisualPlanningInput.model_validate(next(r for r in llm.requests if r.agent_id == "visual").input_payload)
    research = ResearchBundle.model_validate_json(container.artifacts.read(
        container.artifacts.find(task.task_id, "research_bundle").artifact_id))
    assert planning.research == research and planning.plan.title and planning.lesson.sections

    result = visuals_of(task)
    assert result.status == "complete" and result.failures == [] and task.result.warnings == []
    arts = artifacts_by_name(container, task)
    plan_art = arts["visual_plan"]
    assert plan_art.type == ArtifactType.VISUAL_PLAN and plan_art.parent_ids == [arts["research_bundle"].artifact_id]
    plan = VisualPlan.model_validate_json(container.artifacts.read(plan_art.artifact_id))
    assert plan == result.plan and len(plan.requirements) == 4

    # VisualPlan -> IMAGE_ASSET -> source / generation metadata, and the bytes behind each asset.
    images = {a.name: a for a in arts.values() if a.type == ArtifactType.IMAGE_ASSET}
    assert set(images) == {f"image_{r.visual_id}" for r in plan.requirements}
    for art in images.values():
        asset = ImageAsset.model_validate(art.metadata)
        req = plan.requirement(asset.visual_id)
        assert art.parent_ids == [plan_art.artifact_id] and asset.plan_id == plan.plan_id
        assert asset.lesson_section_id == req.lesson_section_id and asset.validation.valid
        data = container.artifacts.read(art.artifact_id)
        probe = probe_image(data)
        assert (probe.width, probe.height, probe.media_type) == (asset.width, asset.height, art.media_type)
        assert art.content_hash == asset.object.checksum and art.uri == asset.object.uri
        if asset.origin == "search":
            assert asset.attribution.kind == "search" and asset.attribution.license is not None
            assert asset.selection is not None and asset.selection.rank >= 1
        else:
            assert asset.attribution.kind == "generated" and asset.selection is None
            assert asset.attribution.prompt == req.generation_prompt
    origins = {ImageAsset.model_validate(a.metadata).visual_id: ImageAsset.model_validate(a.metadata).origin
               for a in images.values()}
    assert origins == {"v1_photo": "search", "v1_diagram": "generated", "v2_illustration": "search",
                       "v2_diagram": "generated"}

    # Each lesson section references the IMAGE_ASSET artifacts that illustrate it; the lesson depends on them.
    lesson = lesson_of(container, task)
    for section in lesson.sections:
        refs = {v.artifact_id for v in section.visuals}
        assert refs == {a.artifact_id for a in result.for_section(section.section_id)} and len(refs) == 2
        assert all(container.artifacts.get(r).type == ArtifactType.IMAGE_ASSET for r in refs)
    assert set(arts["lesson"].parent_ids) >= {a.artifact_id for a in images.values()}
    slides = json.loads(container.artifacts.read(arts["slide_plan"].artifact_id))
    assert slides["slides"]  # slides are planned after the visuals, from the illustrated lesson
    assert {a.artifact_id for a in task.result.artifacts} >= {plan_art.artifact_id, *(a.artifact_id for a in images.values())}


async def test_searched_image_validation_failure_falls_back_to_the_next_candidate(make_container) -> None:
    container = make_container()
    task = await run_lesson(container)
    photo = ImageAsset.model_validate(artifacts_by_name(container, task)["image_v1_photo"].metadata)
    # The top-ranked candidate advertised 1600x900 but served 1200x900: rejected with structured errors.
    assert photo.selection.rank == 2
    rejected = photo.selection.rejected
    assert any(r["stage"] == "validate" and "dimension_mismatch" in r["reason"] for r in rejected)
    failed = [e for e in container.task_service.events(task.task_id) if e.type == "image.validation_failed"]
    assert len(failed) == 1 and {e["code"] for e in failed[0].data["errors"]} == {"dimension_mismatch",
                                                                                   "aspect_ratio_mismatch"}
    # Attribution is the provider's, verbatim.
    catalog = {e["provider_image_id"]: e for e in json.loads(CATALOG.read_text("utf-8"))}
    entry = catalog[photo.attribution.provider_image_id]
    a = photo.attribution
    assert (a.creator, a.publisher, a.source_url, a.image_url, a.license.name, a.license.url, a.attribution_text) == \
        (entry["creator"], entry["publisher"], entry["source_url"], entry["url"], entry["license_name"],
         entry["license_url"], entry["attribution_text"])


async def test_visual_events_are_emitted_on_the_existing_bus(make_container) -> None:
    container = make_container()
    task = await run_lesson(container)
    events = [e for e in container.task_service.events(task.task_id) if e.type.startswith(("visual.", "image."))]
    types = [e.type for e in events]
    assert set(types) == set(VISUAL_EVENTS)
    assert types[0] == "visual.started" and types[1] == "visual.plan_created" and types[-1] == "visual.completed"
    assert all(e.task_id == task.task_id and e.node_id == "visual" and e.agent_id == "visual" for e in events)
    assert types.count("image.asset_created") == 4 and types.count("image.generation_completed") == 2
    created = [e.data for e in events if e.type == "image.asset_created"]
    assert {d["artifact_id"] for d in created} == {a.artifact_id for a in visuals_of(task).assets}
    review_done = max(i for i, e in enumerate(container.task_service.events(task.task_id))
                      if e.type == "review.completed")
    first_visual = next(i for i, e in enumerate(container.task_service.events(task.task_id))
                        if e.type == "visual.started")
    assert review_done < first_visual


async def test_visual_cost_and_usage_tracking(make_container) -> None:
    container = make_container()
    task = await run_lesson(container)
    cost = task.cost
    generation = cost.by_service["image_generation:mock"]
    assert generation.calls == 2 and generation.results == 2 and generation.cost_usd == 0.0
    assert generation.units == {"images": 2.0, "megapixels": 1.8432}
    search = cost.by_service["image_search:mock"]
    assert search.calls == 2 and search.cost_usd == 0.0 and search.units == {"requests": 2.0}
    assert cost.by_service["image_fetch:mock"].cost_usd is None  # not reported, so not invented
    assert cost.by_agent["visual"].calls == 1 and cost.by_agent["visual"].cost_usd > 0
    assert cost.llm_calls == sum(line.calls for line in cost.by_agent.values())
    assert container.orchestrator._planner.estimate(container.orchestrator._planner.template("lesson_generation")) \
        == cost.estimated_cost_usd


async def test_image_metadata_never_enters_the_textual_citation_chain(make_container) -> None:
    container = make_container()
    task = await run_lesson(container)
    research = ResearchBundle.model_validate_json(container.artifacts.read(
        container.artifacts.find(task.task_id, "research_bundle").artifact_id))
    lesson = lesson_of(container, task)
    assert {c.citation_id for c in lesson.references} == {c for s in lesson.sections for c in s.citations}
    assert all(c in research.citations for c in lesson.references)
    image_urls = set()
    for ref in visuals_of(task).assets:
        asset = ImageAsset.model_validate(container.artifacts.get(ref.artifact_id).metadata)
        if asset.attribution.kind == "search":
            image_urls |= {asset.attribution.image_url, asset.attribution.source_url}
        else:
            assert not hasattr(asset.attribution, "source_url")  # generated images are never sourced evidence
    assert image_urls and not image_urls & ({s.url for s in research.sources} | {c.url for c in research.citations})
    for section in lesson.sections:
        for cid in section.citations:
            _, evidence, source = research.resolve(cid)  # still Lesson -> Citation -> Evidence -> Source
            assert evidence.target_id == section.concept_id


async def test_rejected_lesson_drafts_never_create_image_assets(make_container) -> None:
    llm = MockLLMProvider({**default_responders(), "content_reviewer": always_reject})
    container = make_container(llm=llm, max_revisions=1)
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED and task.errors[-1].node_id == "teach_review"
    assert llm.calls["visual"] == 0
    assert task.workflow.node_states["visual"].status.value == "PENDING"
    types = {a.type for a in container.task_service.artifacts(task.task_id)}
    assert ArtifactType.IMAGE_ASSET not in types and ArtifactType.VISUAL_PLAN not in types
    assert not [e for e in container.task_service.events(task.task_id) if e.type.startswith(("visual.", "image."))]


async def test_lesson_accepted_with_warnings_gets_no_visuals(make_container) -> None:
    llm = MockLLMProvider({**default_responders(), "content_reviewer": always_reject})
    container = make_container(llm=llm, max_revisions=1, revision_exhausted_policy="accept_with_warnings")
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    assert task.workflow.node_states["visual"].status.value == "SKIPPED"
    assert task.workflow.node_states["visual_policy"].status.value == "SKIPPED"
    assert llm.calls["visual"] == 0
    assert ArtifactType.IMAGE_ASSET not in {a.type for a in container.task_service.artifacts(task.task_id)}
    assert any("only generated for an approved lesson" in w for w in task.result.warnings)
    assert all(s.visuals == [] for s in lesson_of(container, task).sections)


async def test_visuals_run_once_after_revision_on_the_approved_lesson(make_container) -> None:
    llm = MockLLMProvider(default_responders())  # the first draft is rejected, the revision approved
    container = make_container(llm=llm)
    task = await run_lesson(container)
    assert task.result.revisions == 1 and llm.calls["teacher"] == 2 and llm.calls["visual"] == 1
    planned = VisualPlanningInput.model_validate(next(r for r in llm.requests if r.agent_id == "visual").input_payload)
    review = task.workflow.node_states["teach_review"].output
    approved = LessonContent.model_validate(review["candidate"])
    assert planned.lesson.model_dump(exclude={"references"}) == approved.model_dump(exclude={"references"})
    assert planned.lesson.sections[-1].citations  # the revision, not the rejected draft (which had none)


async def test_optional_visual_failure_continues_with_a_recorded_warning(make_container) -> None:
    # The optional section-2 illustration finds no image, and its generation fallback fails too.
    container = make_container(image_search_provider=SearchWithout("match"),
                               image_generation_provider=GenerationFailing(prefix="Illustration"))
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    result = visuals_of(task)
    assert result.status == "partial" and len(result.assets) == 3
    [failure] = result.failures
    assert failure.visual_id == "v2_illustration" and not failure.required
    assert [(a.source, a.stage) for a in failure.attempts] == [("search", "search"), ("generate", "generate")]
    assert "image generation API unavailable" in failure.reason
    warning = next(w for w in task.result.warnings if "v2_illustration" in w)
    assert warning.startswith("Optional visual")
    lesson_art = artifacts_by_name(container, task)["lesson"]
    assert lesson_art.metadata["visual_status"] == "partial" and lesson_art.metadata["visual_warnings"] == [warning]
    assert "image_v2_illustration" not in artifacts_by_name(container, task)
    completed = next(e for e in container.task_service.events(task.task_id) if e.type == "visual.completed")
    assert completed.data["failures"] == [{"visual_id": "v2_illustration", "required": False,
                                           "reason": failure.reason}]


async def test_optional_visual_falls_back_from_search_to_generation(make_container) -> None:
    container = make_container(image_search_provider=SearchWithout("match", fail=True))
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED and task.result.warnings == []
    illustration = ImageAsset.model_validate(artifacts_by_name(container, task)["image_v2_illustration"].metadata)
    assert illustration.origin == "generated" and illustration.attribution.kind == "generated"


async def test_required_visual_failure_fails_the_task_by_default(make_container) -> None:
    container = make_container(image_generation_provider=GenerationFailing())
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED
    assert task.errors[-1].node_id == "visual_policy"
    assert "required visuals failed" in task.errors[-1].message and "v1_diagram" in task.errors[-1].message
    result = VisualResult.model_validate(task.workflow.node_states["visual"].output)
    assert result.status == "failed" and {f.visual_id for f in result.failures} == {"v1_diagram", "v2_diagram"}
    assert "visual.failed" in [e.type for e in container.task_service.events(task.task_id)]
    names = set(artifacts_by_name(container, task))
    assert "lesson" not in names and "visual_plan" in names  # the attempt is recorded; no lesson was stored


async def test_required_visual_failure_can_continue_with_a_warning(make_container) -> None:
    container = make_container(image_generation_provider=GenerationFailing(), visual_failure_policy="continue")
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    assert any("Required visuals failed" in w and "v2_diagram" in w for w in task.result.warnings)
    lesson = lesson_of(container, task)
    assert {v.visual_id for s in lesson.sections for v in s.visuals} == {"v1_photo", "v2_illustration"}


async def test_photos_are_never_replaced_by_generated_images(make_container) -> None:
    container = make_container(image_search_provider=SearchWithout("", fail=True))
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED and "v1_photo" in task.errors[-1].message
    result = VisualResult.model_validate(task.workflow.node_states["visual"].output)
    photo = next(f for f in result.failures if f.visual_id == "v1_photo")
    assert [a.source for a in photo.attempts] == ["search"]
    # The optional illustration may fall back to generation; the photo may not.
    assert "v2_illustration" in {a.visual_id for a in result.assets}


async def test_transient_generation_failure_is_retried_without_duplicates(make_container) -> None:
    flaky = GenerationFailing(transient_once=True)
    container = make_container(image_generation_provider=flaky)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    assert flaky.calls == 3  # one retried failure, then two successful generations
    tool_failures = [e for e in container.task_service.events(task.task_id)
                     if e.type == "tool.failed" and e.tool == "image.generate"]
    assert len(tool_failures) == 1 and len(visuals_of(task).assets) == 4


async def test_identical_images_are_stored_once_across_tasks(make_container, tmp_path) -> None:
    container = make_container()
    first = await run_lesson(container)
    objects = tmp_path / "d0" / "objects" / "objects"
    stored = sorted(p.name for p in objects.rglob("*.png"))
    # A second learner with the same profile and answers gets the same lesson, so the same images.
    container.learner_service.upsert("learner-2", LearnerProfileInput.model_validate(LEARNER["profile"]))
    second = await container.task_service.create_and_run(request=LEARNER["request"], learner_id="learner-2",
                                                         user_id="u2")
    while second.status == TaskStatus.WAITING:
        second = await container.task_service.submit_assessment(second.task_id, answers_for(second))
    assert second.status == TaskStatus.COMPLETED, second.errors
    a, b = visuals_of(first), visuals_of(second)
    assert [(x.visual_id, x.checksum, x.uri) for x in a.assets] == [(x.visual_id, x.checksum, x.uri) for x in b.assets]
    assert {x.artifact_id for x in a.assets}.isdisjoint({x.artifact_id for x in b.assets})  # per-task artifacts
    assert sorted(p.name for p in objects.rglob("*.png")) == stored  # no object was written twice
    reused = [e.data["reused_object"] for e in container.task_service.events(second.task_id)
              if e.type == "image.generation_completed"]
    assert reused == [True, True]


async def test_crash_after_visuals_resumes_without_regenerating(make_container, tmp_path) -> None:
    data_dir = tmp_path / "shared"
    first = make_container(observers=[crash_after("visual_policy")], data_dir=data_dir)
    learner_id = add_demo_learner(first)
    task = await first.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                   learner_id=learner_id, user_id="u1")
    task = await first.task_service.submit_assessment(task.task_id, answers_for(task))
    with pytest.raises(SimulatedCrash):
        await first.task_service.submit_assessment(task.task_id, answers_for(task))
    before = {a.artifact_id for a in first.task_service.artifacts(task.task_id)}

    llm = MockLLMProvider(default_responders())
    generation = GenerationFailing(prefix="")  # any generation now would fail the task
    second = make_container(llm=llm, data_dir=data_dir, image_generation_provider=generation)
    resumed = await second.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors
    assert llm.calls["visual"] == 0 and generation.calls == 0
    events = second.task_service.events(task.task_id)
    assert sum(1 for e in events if e.type == "visual.started") == 1
    images = {a.artifact_id for a in second.task_service.artifacts(task.task_id) if a.type == ArtifactType.IMAGE_ASSET}
    assert len(images) == 4 and images <= before
    lesson = lesson_of(second, resumed)
    assert {v.artifact_id for s in lesson.sections for v in s.visuals} == images


async def test_visual_planner_output_is_validated_and_corrected(make_container) -> None:
    llm = MockLLMProvider(default_responders())
    bad_url = {"requirements": [{
        "visual_id": "v1", "purpose": "p", "lesson_section_id": "sec_es.football.opinions", "concept": "c",
        "description": "use https://example.org/cat.png", "visual_type": "photo", "preferred_source": "search",
        "search_query": "fans"}]}
    unknown_section = {"requirements": [{**bad_url["requirements"][0], "description": "fans",
                                         "lesson_section_id": "sec_invented"}]}
    llm.inject("visual", json.dumps(bad_url), json.dumps(unknown_section))
    container = make_container(llm=llm)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    assert llm.calls["visual"] == 3  # two rejected proposals, then a valid plan
    failures = [e.data["error"] for e in container.task_service.events(task.task_id)
                if e.type == "agent.validation_failed" and e.agent_id == "visual"]
    assert "URL" in failures[0] and "unknown lesson sections" in failures[1]
    assert len(visuals_of(task).plan.requirements) == 4


async def test_visuals_can_be_disabled(make_container) -> None:
    llm = MockLLMProvider(default_responders())
    container = make_container(llm=llm, visual_max_per_lesson=0)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED and llm.calls["visual"] == 0
    result = visuals_of(task)
    assert result.status == "complete" and result.plan.requirements == [] and result.assets == []
