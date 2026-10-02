"""Production runs end to end, on the real adapters over in-process fakes of the vendor APIs (no network, no
credentials): configuration, dry run, budgets, the artifact graph, traceability, the report, resume, repair,
idempotency, failure classification and evaluation. The workflow, agents and tools are the ordinary ones."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config.production import ProductionSettings
from app.config.providers import ProviderSettings
from app.config.settings import Settings
from app.providers.image.openai import OpenAIImageGenerationProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.search.tavily import TavilySearchProvider
from app.providers.tts.openai import OpenAITTSProvider
from app.schemas.artifact import ArtifactType
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer
from app.schemas.production import REQUIRED_CHAIN, ProductionTask
from app.schemas.task import TaskStatus
from app.services.container import build_container
from app.services.production import RunOutcome
from tests.conftest import FIXTURES
from tests.fake_vendors import FAKE_KEY, FakeChatModel, FakeOpenAI, FakeTavily, http_client

SEARCH_KEY = "tvly-test-0123456789abcdef"
TASK = ProductionTask(level="B1", topic="Climate change", language="es", learner_id="prod-learner")
DIAGNOSTIC_KEY = {"es.climate.vocabulary": "el calentamiento global", "es.climate.cause_effect": "por eso",
                  "es.climate.necessity_subjunctive": ""}


class SimulatedCrash(BaseException):
    """The process dying: not an Exception, so nothing in the runtime catches it."""


def crash_after(node_id: str):
    def observer(task, finished: str) -> None:
        if finished == node_id:
            raise SimulatedCrash(node_id)
    return observer


async def answer(sheet: DiagnosticQuestionSheet) -> DiagnosticAnswers:
    return DiagnosticAnswers(answers=[LearnerAnswer(question_id=q.question_id, answer=DIAGNOSTIC_KEY.get(q.concept_id, ""))
                                      for q in sheet.questions])


def production_settings(tmp_path, *, budget: dict | None = None, **providers) -> Settings:
    values = dict(teaching_agent_mode="production", teaching_agent_offline=False, llm_routes={}, llm_provider="openai", llm_model="gpt-test",
                  openai_api_key=FAKE_KEY, tts_provider="openai", image_provider="openai", search_provider="tavily",
                  search_api_key=SEARCH_KEY, tts_languages="es-ES,en-US")
    values.update(providers)
    return Settings(data_dir=tmp_path / "data", log_json=False, providers=ProviderSettings(**values),
                    production=ProductionSettings(**(budget or {})))


class Vendors:
    """The fake OpenAI and Tavily APIs, and real adapters pointed at them."""

    def __init__(self) -> None:
        self.chat = FakeChatModel(default_responders())
        self.openai = FakeOpenAI(self.chat)
        self.tavily = FakeTavily(FIXTURES / "web_corpus.json")

    def container(self, settings: Settings, **kwargs):
        base = "https://api.openai.example/v1"
        llm = OpenAICompatibleLLMProvider(http_client("openai", base, self.openai,
                                                      request_id_header="X-Client-Request-Id"))
        tts = OpenAITTSProvider(http_client("openai", base, self.openai), languages=["es-ES", "en-US"])
        images = OpenAIImageGenerationProvider(http_client("openai", base, self.openai))
        search = TavilySearchProvider(http_client("tavily", "https://api.tavily.example", self.tavily,
                                                  headers={"Authorization": f"Bearer {SEARCH_KEY}"}))
        c = build_container(settings, llm_providers={"openai": llm}, tts_provider=tts,
                            image_generation_provider=images, search_provider=search, **kwargs)
        self.chat.prompts.update({c.agents.get(a.id).system_prompt: a.id for a in c.agents.describe()})
        return c

    def count(self, path: str) -> int:
        return sum(1 for r in self.openai.requests if r.url.path == path)


@pytest.fixture
def online(monkeypatch):
    """The suite runs with TEACHING_AGENT_OFFLINE=true; these tests use real adapters over fakes instead."""
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)


@pytest.fixture
def vendors(online) -> Vendors:
    return Vendors()


def requests_by_node(task) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in task.cost.provider_requests:
        counts[r.node_id] = counts.get(r.node_id, 0) + 1
    return counts


# --- configuration and dry run -------------------------------------------------------------------------------


async def test_offline_plan_is_not_ready_and_names_what_to_set(container) -> None:
    plan = await container.production.plan(TASK)
    assert plan.mode == "offline" and not plan.ready
    text = "\n".join(plan.problems)
    assert "TEACHING_AGENT_MODE=production" in text
    for var in ("LLM_PROVIDER", "SEARCH_PROVIDER", "IMAGE_PROVIDER", "TTS_PROVIDER"):
        assert var in text, var
    assert plan.lesson_request.subject == "spanish" and plan.lesson_request.framework_id == "cefr"
    assert plan.lesson_request.language_of_instruction == "es" and plan.knowledge_concepts == 3
    assert plan.stages[0] == "learner_snapshot" and "compose_video" in plan.stages


async def test_dry_run_plan_makes_no_provider_call(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path))
    try:
        plan = await c.production.plan(TASK)
        assert plan.ready, plan.problems
        assert plan.mode == "production" and set(plan.required_capabilities) == {"llm", "search", "image", "tts"}
        assert {p.capability: (p.provider, p.real) for p in plan.providers if p.required} == {
            "llm": ("openai", True), "search": ("tavily", True), "image": ("openai", True), "tts": ("openai", True)}
        assert plan.estimated_llm_cost_usd is None  # gpt-test has no configured price: unknown, not invented
        assert plan.budget.max_searched_images == 0 and plan.budget.max_generated_images == 4
        assert vendors.openai.requests == [] and vendors.tavily.requests == []
        assert list(c.providers.invoker.usage_log) == []
    finally:
        c.close()


async def test_capabilities_the_run_does_not_use_are_not_required(vendors, tmp_path) -> None:
    settings = production_settings(tmp_path, image_provider="mock", budget={"max_generated_images": 0})
    c = build_container(settings)
    try:
        plan = await c.production.plan(TASK)
        assert "image" not in plan.required_capabilities
        assert not any("IMAGE_PROVIDER" in p for p in plan.problems)
    finally:
        c.close()


async def test_health_checks_only_the_required_capabilities(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path, budget={"max_generated_images": 0}))
    try:
        plan = await c.production.plan(TASK)
        statuses = await c.production.check_health(plan.required_capabilities)
        assert {s.capability.value for s in statuses} == {"llm", "search", "tts"}
        assert all(s.available for s in statuses)
    finally:
        c.close()


# --- the production run ---------------------------------------------------------------------------------------


async def test_production_run_produces_the_artifact_graph_trace_and_report(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path))
    try:
        plan = await c.production.plan(TASK)
        outcome = await c.production.run(plan, answer=answer)
        task = outcome.task
        assert task.status == TaskStatus.COMPLETED, task.errors
        assert task.metadata["budget"]["max_llm_requests"] == 80  # the budget is visible on the task

        report = c.production.report(outcome, plan, start_time=task.created_at)
        assert report.status == "COMPLETED" and report.graph_check.complete, report.graph_check
        types = {a.type for a in report.artifact_graph}
        assert set(REQUIRED_CHAIN) <= types
        by_id = {a.artifact_id: a for a in report.artifact_graph}
        assert all(p in by_id for a in report.artifact_graph for p in a.parent_ids)  # parents preserved, not flattened
        video = next(a for a in report.artifact_graph if a.type == ArtifactType.VIDEO)
        assert report.final_video is not None and report.final_video.artifact_id == video.artifact_id
        assert {"PRESENTATION_TIMELINE", "PRESENTATION", "AUDIO_ASSET", "IMAGE_ASSET", "LESSON", "LESSON_PLAN",
                "RESEARCH_BUNDLE"} <= set(report.graph_check.video_ancestor_types)
        images = [a for a in report.artifact_graph if a.type == ArtifactType.IMAGE_ASSET]
        assert images and all(a.provider == "openai" or a.provider == "teaching-agent" for a in images)

        # Traceability: task -> node -> provider request -> artifact.
        requests = task.cost.provider_requests
        assert requests and all(r.node_id for r in requests)
        assert len({r.request_id for r in requests}) == len(requests)
        chat_ids = {r.headers["X-Client-Request-Id"] for r in vendors.openai.requests
                    if r.url.path == "/v1/chat/completions"}
        assert chat_ids == {r.request_id for r in requests if r.capability == "llm"}
        image_node = next(n for n in report.trace if n.node_id == "visual")
        assert image_node.request_ids and {a.artifact_id for a in images} <= set(image_node.artifact_ids)
        assert all(a.node_id for a in report.artifact_graph)

        usage = report.usage
        assert usage.llm_requests == vendors.count("/v1/chat/completions")
        assert usage.image_generations == vendors.count("/v1/images/generations") == len(images)
        assert usage.tts_requests == vendors.count("/v1/audio/speech")
        assert usage.search_requests == len(vendors.tavily.requests)
        assert usage.tts_characters > 0 and usage.tts_seconds > 0 and usage.input_tokens > 0
        assert not report.cost_complete and "llm:openai/gpt-test" in usage.unpriced  # no price: never invented
        assert report.summary["visual"]["searched"] == 0  # MAX_SEARCHED_IMAGES=0: generated visuals only
    finally:
        c.close()


async def test_identical_run_is_reused_and_a_changed_configuration_is_not(vendors, tmp_path) -> None:
    settings = production_settings(tmp_path)
    c = vendors.container(settings)
    try:
        plan = await c.production.plan(TASK)
        first = await c.production.run(plan, answer=answer)
        assert first.task.status == TaskStatus.COMPLETED
        sent = len(vendors.openai.requests) + len(vendors.tavily.requests)
        again = await c.production.run(await c.production.plan(TASK), answer=answer)
        assert again.reused and again.task.task_id == first.task.task_id
        assert len(vendors.openai.requests) + len(vendors.tavily.requests) == sent  # nothing regenerated
    finally:
        c.close()
    other = vendors.container(production_settings(tmp_path, tts_model="tts-other"))
    try:
        assert (await other.production.plan(TASK)).run_key != plan.run_key
        same = vendors.container(settings)
        try:
            assert (await same.production.plan(TASK)).run_key == plan.run_key  # deterministic across processes
        finally:
            same.close()
    finally:
        other.close()


async def test_kill_after_presentation_then_resume_reuses_everything_before(vendors, tmp_path) -> None:
    settings = production_settings(tmp_path)
    first = vendors.container(settings, observers=[crash_after("render_presentation")])
    plan = await first.production.plan(TASK)
    with pytest.raises(SimulatedCrash):
        await first.production.run(plan, answer=answer)
    crashed = first.task_service.list_for_learner(TASK.learner_id)[-1]
    first.close()
    before = requests_by_node(crashed)
    images_before = vendors.count("/v1/images/generations")
    pptx = [a for a in first.artifacts.list_for_task(crashed.task_id) if a.type == ArtifactType.PRESENTATION]
    assert len(pptx) == 1 and before.get("visual") and before.get("research")

    second = vendors.container(settings)
    try:
        outcome = await second.production.run(await second.production.plan(TASK), answer=answer)
        assert outcome.resumed and outcome.task.task_id == crashed.task_id
        assert outcome.task.status == TaskStatus.COMPLETED, outcome.task.errors
        after = requests_by_node(outcome.task)
        for node in ("research", "plan", "teach_review", "visual", "slide_plan", "diagnose_1"):
            assert after.get(node, 0) == before.get(node, 0), node  # not repeated
        assert vendors.count("/v1/images/generations") == images_before  # images not regenerated
        assert after.get("synthesize_audio")  # the remaining stages ran
        artifacts = second.artifacts.list_for_task(crashed.task_id)
        assert [a.artifact_id for a in artifacts if a.type == ArtifactType.PRESENTATION] == [pptx[0].artifact_id]
        events = second.task_service.events(crashed.task_id)
        assert sum(1 for e in events if e.type == "node.started" and e.node_id == "render_presentation") == 1
    finally:
        second.close()


async def test_resume_regenerates_only_what_is_missing_or_corrupt(vendors, tmp_path) -> None:
    settings = production_settings(tmp_path)
    first = vendors.container(settings, observers=[crash_after("render_presentation")])
    plan = await first.production.plan(TASK)
    with pytest.raises(SimulatedCrash):
        await first.production.run(plan, answer=answer)
    crashed = first.task_service.list_for_learner(TASK.learner_id)[-1]
    image = next(a for a in first.artifacts.list_for_task(crashed.task_id) if a.type == ArtifactType.IMAGE_ASSET)
    first.close()
    path = image.uri.removeprefix("file://")
    with open(path, "wb") as fh:
        fh.write(b"corrupted")
    research_before = requests_by_node(crashed).get("research")

    second = vendors.container(settings)
    try:
        outcome = await second.production.resume(crashed.task_id, answer=answer)
        assert outcome.task.status == TaskStatus.COMPLETED, outcome.task.errors
        assert "visual" in outcome.repaired_nodes and "research" not in outcome.repaired_nodes
        assert any("repaired" in w for w in outcome.warnings)
        assert requests_by_node(outcome.task).get("research") == research_before
        for artifact in second.artifacts.list_for_task(crashed.task_id):
            if artifact.type == ArtifactType.IMAGE_ASSET and artifact.content_hash == image.content_hash:
                assert second.artifacts.verify(artifact) is None  # rewritten intact
    finally:
        second.close()


# --- budgets --------------------------------------------------------------------------------------------------


async def test_llm_request_budget_stops_the_task_before_exceeding_it(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path, budget={"max_llm_requests": 3}))
    try:
        outcome = await c.production.run(await c.production.plan(TASK), answer=answer)
        task = outcome.task
        assert task.status == TaskStatus.FAILED
        error = task.errors[-1]
        assert error.category == "BudgetExceededError" and "MAX_LLM_REQUESTS" in error.message
        assert vendors.count("/v1/chat/completions") == 3  # never a fourth request
    finally:
        c.close()


async def test_tts_character_budget_is_checked_before_synthesis(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path, budget={"max_tts_characters": 60}))
    try:
        task = (await c.production.run(await c.production.plan(TASK), answer=answer)).task
        assert task.status == TaskStatus.FAILED
        assert task.errors[-1].category == "BudgetExceededError" and task.errors[-1].stage == "AudioError"
        sent = sum(len(json.loads(r.content)["input"]) for r in vendors.openai.requests
                   if r.url.path == "/v1/audio/speech")
        assert sent <= 60
        assert not any(a.type == ArtifactType.VIDEO for a in c.artifacts.list_for_task(task.task_id))
    finally:
        c.close()


async def test_image_budget_shapes_the_visual_plan(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path, budget={"max_generated_images": 1}))
    try:
        outcome = await c.production.run(await c.production.plan(TASK), answer=answer)
        assert outcome.task.status == TaskStatus.COMPLETED, outcome.task.errors
        assert vendors.count("/v1/images/generations") == 1
        plan_artifact = next(a for a in c.artifacts.list_for_task(outcome.task.task_id)
                             if a.type == ArtifactType.VISUAL_PLAN)
        plan = json.loads(c.artifacts.read(plan_artifact.artifact_id))
        assert len(plan["requirements"]) == 1 and plan["requirements"][0]["preferred_source"] == "generate"
    finally:
        c.close()


async def test_cost_limit_aborts_once_reported_usage_exceeds_it(vendors, tmp_path) -> None:
    settings = production_settings(tmp_path, llm_input_price_per_mtok=1000.0, llm_output_price_per_mtok=1000.0,
                                   budget={"max_cost_usd": 5.0})
    c = vendors.container(settings)
    try:
        task = (await c.production.run(await c.production.plan(TASK), answer=answer)).task
        assert task.status == TaskStatus.FAILED and task.errors[-1].category == "BudgetExceededError"
        assert "MAX_COST_USD" in task.errors[-1].message
        usage = c.production.report(RunOutcome(task=task), await c.production.plan(TASK),
                                    start_time=task.created_at).usage
        # It stopped at the request that crossed the limit: at most one request's worth over it.
        last = max(r.estimated_cost_usd or 0 for r in task.cost.provider_requests)
        assert 5.0 < usage.estimated_cost_usd <= 5.0 + last
    finally:
        c.close()


# --- failures -------------------------------------------------------------------------------------------------


async def test_search_outage_degrades_to_the_knowledge_base_with_sanitized_warnings(vendors, tmp_path) -> None:
    """The research policy fails a task only when research produced nothing; a web search outage leaves the
    knowledge-base sources, and every warning names the failure without the credential."""
    c = vendors.container(production_settings(tmp_path))
    vendors.tavily.fail = [httpx.Response(401, json={"detail": f"invalid key {SEARCH_KEY}"})] * 20
    try:
        task = (await c.production.run(await c.production.plan(TASK), answer=answer)).task
        assert task.status == TaskStatus.COMPLETED, task.errors
        assert any("tavily: HTTP 401" in w for w in task.result.warnings)
        stored = task.model_dump_json() + json.dumps([e.model_dump(mode="json")
                                                       for e in c.task_service.events(task.task_id)])
        assert SEARCH_KEY not in stored and FAKE_KEY not in stored
    finally:
        c.close()


async def test_provider_authentication_failure_is_a_provider_error(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path))
    vendors.openai.fail = [httpx.Response(401, json={"error": {"message": f"bad key {FAKE_KEY}"}})]
    try:
        task = (await c.production.run(await c.production.plan(TASK), answer=answer)).task
        assert task.status == TaskStatus.FAILED
        assert task.errors[-1].category == "ProviderError" and task.errors[-1].stage == "PlanningError"
        assert FAKE_KEY not in task.model_dump_json()
    finally:
        c.close()


# --- evaluation -----------------------------------------------------------------------------------------------


async def test_the_generated_lesson_stays_evaluable(vendors, tmp_path) -> None:
    c = vendors.container(production_settings(tmp_path))
    try:
        lesson = (await c.production.run(await c.production.plan(TASK), answer=answer)).task
        assert lesson.status == TaskStatus.COMPLETED
        taken = c.learner_service.progress(TASK.learner_id).assessments_taken
        evaluation = await c.production.start_evaluation(lesson.task_id)
        assert evaluation.status == TaskStatus.WAITING and evaluation.waiting.kind == "assessment_answers"
        assert c.production.evaluation_for(lesson.task_id).task_id == evaluation.task_id
        payload = {"answers": [{"question_id": q["question_id"], "answer": (q.get("choices") or [""])[0]}
                               for q in evaluation.waiting.prompt["questions"]]}
        done = await c.production.submit_evaluation(evaluation.task_id, payload)
        assert done.status == TaskStatus.COMPLETED, done.errors
        assert c.learner_service.progress(TASK.learner_id).assessments_taken == taken + 1  # learner memory updated
    finally:
        c.close()
