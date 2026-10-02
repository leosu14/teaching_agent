"""Optional live end-to-end test: one small production lesson on the real configured providers.

Skipped unless RUN_LIVE_E2E=true and production mode is configured with credentials. It spends real money, bounded by
LIVE_E2E_MAX_COST (USD, default 1.00) on top of tight unit limits; CI never runs it.

    RUN_LIVE_E2E=true LIVE_E2E_MAX_COST=0.50 TEACHING_AGENT_MODE=production LLM_PROVIDER=openai LLM_MODEL=... \\
        OPENAI_API_KEY=... TTS_PROVIDER=openai IMAGE_PROVIDER=openai SEARCH_PROVIDER=tavily TAVILY_API_KEY=... \\
        pytest tests/live -s
"""

from __future__ import annotations

import os

import pytest

from app.config.production import ProductionSettings
from app.config.settings import Settings
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer
from app.schemas.production import ProductionTask
from app.services.container import build_container

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_E2E", "").lower() != "true",
                                reason="set RUN_LIVE_E2E=true (and production credentials) to run the live E2E test")

SECRET_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TTS_API_KEY", "IMAGE_API_KEY", "SEARCH_API_KEY",
               "TAVILY_API_KEY")


async def unanswered(sheet: DiagnosticQuestionSheet) -> DiagnosticAnswers:
    return DiagnosticAnswers(answers=[LearnerAnswer(question_id=q.question_id, answer="") for q in sheet.questions])


async def test_live_production_lesson(tmp_path) -> None:
    max_cost = float(os.environ.get("LIVE_E2E_MAX_COST") or 1.0)
    production = ProductionSettings(max_cost_usd=max_cost, max_llm_requests=40, max_llm_tokens=300_000,
                                    max_search_requests=8, max_generated_images=1, max_searched_images=0,
                                    max_tts_characters=8_000, max_tts_seconds=900)
    settings = Settings(data_dir=tmp_path / "data", log_json=False, production=production)
    if settings.providers.offline:
        pytest.skip("offline mode: set TEACHING_AGENT_MODE=production (and not TEACHING_AGENT_OFFLINE=true)")
    c = build_container(settings)
    try:
        plan = await c.production.plan(ProductionTask(level="B1", topic="Climate change", language="es",
                                                      learner_id="live-e2e"))
        assert plan.ready, plan.problems
        if plan.estimated_llm_cost_usd is not None and plan.estimated_llm_cost_usd > max_cost:
            pytest.fail(f"estimated LLM cost ${plan.estimated_llm_cost_usd:.4f} exceeds LIVE_E2E_MAX_COST={max_cost}")
        health = await c.production.check_health(plan.required_capabilities)
        assert all(h.available for h in health), [h.model_dump() for h in health if not h.available]

        outcome = await c.production.run(plan, answer=unanswered, fresh=True)
        report = c.production.report(outcome, plan, start_time=outcome.task.created_at, health=health)
        c.production.export(report, tmp_path / "out")
        (tmp_path / "out" / "production_run.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nlive E2E: {report.status}, cost ${report.estimated_cost_usd:.4f} "
              f"(complete={report.cost_complete}), report {tmp_path / 'out' / 'production_run.json'}")

        assert report.estimated_cost_usd <= max_cost + 1e-9 or report.status == "FAILED"
        assert report.status == "COMPLETED", report.errors
        assert report.graph_check.complete, report.graph_check
        assert report.final_video is not None and report.final_video.duration and report.final_video.duration > 0
        text = report.model_dump_json()
        for var in SECRET_VARS:
            value = os.environ.get(var)
            if value:
                assert value not in text, f"{var} leaked into the report"
    finally:
        c.close()
