"""One effective task budget (PR19 review F2): a task without a budget of its own gets the default task budget (the
production MAX_* limits), and the visual and video planners plan within that same budget the spend ledger
enforces. A restrictive default shapes the plan instead of failing the lesson at the request the ledger refuses."""

from __future__ import annotations

from app.config.production import ProductionSettings
from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.video_generation.mock import MockVideoGenerationProvider
from app.schemas.artifact import ArtifactType
from app.schemas.generative_video import VideoSegmentPlanSet
from app.schemas.task import TaskStatus
from app.schemas.usage import BUDGET_KEY
from app.schemas.visual import VisualPlan, VisualResult
from app.services.container import build_container
from tests.conftest import run_lesson
from tests.integration.test_visual_flow import GenerationFailing
from tests.integration.test_generative_video_flow import FIXTURES, arts, finish, generation, start


def generated(plan: VisualPlan) -> int:
    return sum(1 for r in plan.requirements if r.preferred_source == "generate")


async def test_a_default_image_limit_shapes_the_visual_plan_of_a_non_production_lesson(tmp_path) -> None:
    c = build_container(Settings(data_dir=tmp_path / "d", log_json=False,
                                 production=ProductionSettings(max_generated_images=1)),
                        llm_providers={"mock": MockLLMProvider(default_responders())})
    try:
        task = await run_lesson(c)
        assert BUDGET_KEY not in task.metadata  # not a production task: the default task budget applies
        assert task.status == TaskStatus.COMPLETED, task.errors
        plan_art = c.artifacts.find(task.task_id, "visual_plan")
        plan = VisualPlan.model_validate_json(c.artifacts.read(plan_art.artifact_id))
        assert generated(plan) <= 1
        images = [r for r in task.cost.provider_requests if r.capability == "image" and r.operation == "generate"]
        assert len(images) <= 1
        assert not any(e.category == "BudgetExceededError" for e in task.errors)
    finally:
        c.close()


async def test_an_explicit_task_budget_still_overrides_the_default(tmp_path) -> None:
    """A task's own budget (a production run's metadata) wins over the default, for the planner too."""
    c = build_container(Settings(data_dir=tmp_path / "d", log_json=False,
                                 production=ProductionSettings(max_generated_images=0)),
                        llm_providers={"mock": MockLLMProvider(default_responders())})
    try:
        assert c.orchestrator._budgets.for_task({BUDGET_KEY: {"max_generated_images": 3}}).max_generated_images == 3
        assert c.orchestrator._budgets.for_task({}).max_generated_images == 0
    finally:
        c.close()


async def test_a_default_video_segment_limit_shapes_the_segment_plan(tmp_path) -> None:
    """Coverage, not a regression: the video settings already carried MAX_GENERATED_VIDEO_SEGMENTS into every lesson's
    segment plan before PR19; the effective budget now tightens the same plan and must keep it within the limit."""
    provider = MockVideoGenerationProvider()
    c = build_container(Settings(data_dir=tmp_path / "d", corpus_dir=FIXTURES, log_json=False, video_composer="mock",
                                 generated_video_poll_interval_seconds=0.0,
                                 production=ProductionSettings(max_generated_video_segments=1)),
                        video_generation_provider=provider)
    try:
        task = await finish(c, await start(c))
        assert task.status == TaskStatus.COMPLETED, task.errors
        a = arts(c, task)
        assert a["video_segment_plan"].type == ArtifactType.VIDEO_SEGMENT_PLAN
        plan = VideoSegmentPlanSet.model_validate_json(c.artifacts.read(a["video_segment_plan"].artifact_id))
        assert len(plan.segments) == 1  # two without the limit (see test_generative_video_flow)
        result = generation(task)
        assert result.status == "complete" and not result.failures and provider.submits == 1
    finally:
        c.close()


def flaky_lesson_container(tmp_path, flaky, **settings):
    return build_container(Settings(data_dir=tmp_path / "d", log_json=False,
                                    production=ProductionSettings(max_generated_images=1), **settings),
                           llm_providers={"mock": MockLLMProvider(default_responders())},
                           image_generation_provider=flaky)


def image_attempts(task) -> list:
    return [r for r in task.cost.provider_requests if r.capability == "image" and r.operation == "generate"]


async def test_a_retry_the_image_budget_refuses_fails_that_visual_not_the_lesson(tmp_path) -> None:
    """PR19 review N1: the plan uses the whole image budget, the one generation fails transiently, and the ledger
    refuses the retry (the failed attempt may have been billed and keeps its charge). The refusal is that visual's
    failure, as for a generated video segment, so the visual failure policy decides; nothing more is sent."""
    flaky = GenerationFailing(transient_once=True)
    c = flaky_lesson_container(tmp_path, flaky, visual_failure_policy="continue")
    try:
        task = await run_lesson(c)
        assert task.status == TaskStatus.COMPLETED, task.errors
        assert not any(e.category == "BudgetExceededError" for e in task.errors)
        assert flaky.calls == 1 and len(image_attempts(task)) == 1  # no paid call after the budget is used up
        result = VisualResult.model_validate(task.workflow.node_states["visual_policy"].output)
        [failure] = [f for f in result.failures if any(a.stage == "budget" for a in f.attempts)]
        assert [(a.source, a.stage) for a in failure.attempts][-1] == ("generate", "budget")
        assert "MAX_GENERATED_IMAGES" in failure.reason
        assert any(failure.visual_id in w for w in task.result.warnings)
        # The failed attempt keeps its charge (no refund to make room for the retry); the retry was never reserved.
        [entry] = [e for e in c.spend.entries(task_id=task.task_id) if e.capability == "image"]
        assert entry.charged.get("generated_images") == 1
    finally:
        c.close()


async def test_a_required_visual_the_image_budget_cannot_retry_follows_the_failure_policy(tmp_path) -> None:
    """With the default policy a required visual that could not be produced fails the lesson as a visual failure
    (as when the provider fails for good), not as a budget stop; still no paid call past the budget."""
    flaky = GenerationFailing(transient_once=True)
    c = flaky_lesson_container(tmp_path, flaky)
    try:
        task = await run_lesson(c)
        assert task.status == TaskStatus.FAILED
        assert task.errors[-1].node_id == "visual_policy" and "required visuals failed" in task.errors[-1].message
        assert task.errors[-1].category != "BudgetExceededError"
        assert flaky.calls == 1 and len(image_attempts(task)) == 1
    finally:
        c.close()


class GenerationBroken(MockImageGenerationProvider):
    """A bug in the provider, not a provider failure or a budget refusal."""

    async def generate(self, request):
        raise RuntimeError("bug in the image adapter")


async def test_an_unrelated_image_error_still_fails_the_visual_step(tmp_path) -> None:
    c = flaky_lesson_container(tmp_path, GenerationBroken(), visual_failure_policy="continue")
    try:
        task = await run_lesson(c)
        assert task.status == TaskStatus.FAILED
        assert task.errors[-1].node_id == "visual" and "bug in the image adapter" in task.errors[-1].message
    finally:
        c.close()
