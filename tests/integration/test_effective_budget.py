"""One effective task budget (PR19 review F2): a task without a budget of its own gets the default task budget (the
production MAX_* limits), and the visual and video planners plan within that same budget the spend ledger
enforces. A restrictive default shapes the plan instead of failing the lesson at the request the ledger refuses."""

from __future__ import annotations

from app.config.production import ProductionSettings
from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.video_generation.mock import MockVideoGenerationProvider
from app.schemas.artifact import ArtifactType
from app.schemas.generative_video import VideoSegmentPlanSet
from app.schemas.task import TaskStatus
from app.schemas.usage import BUDGET_KEY
from app.schemas.visual import VisualPlan
from app.services.container import build_container
from tests.conftest import run_lesson
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
