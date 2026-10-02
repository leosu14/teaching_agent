"""Production configuration, budgets, usage accounting, failure classification and production logging."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from app.agents.base import OutputRejected
from app.agents.visual.agent import VisualAgent
from app.config.production import ProductionSettings, production_readiness, required_capabilities
from app.config.providers import ProviderSettings
from app.config.routing import ConfigError
from app.observability.budget import BudgetExceededError, admit, check_after
from app.observability.events import EventBus
from app.observability.logging import ProductionLogWriter, production_record
from app.observability.redaction import register_secret
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.providers.core.errors import ProviderAuthenticationError, ProviderUnavailable
from app.providers.core.invoker import ProviderInvoker
from app.providers.managed import ManagedTTSProvider
from app.providers.tts.base import ProviderSpeechRequest
from app.providers.tts.mock import MockTTSProvider
from app.runtime.failures import FailureCategory, classify, stage_of
from app.runtime.workflow.nodes import ReviewRejected
from app.runtime.workflows.lesson_generation import PROVIDER_CAPABILITIES
from app.runtime.workflows.research_policy import ResearchRequired
from app.schemas.common import CostSummary, ProviderRequestRecord
from app.schemas.lesson import VisualPlanningInput
from app.schemas.providers import Capability, ProviderPolicy
from app.schemas.usage import TaskBudget, TaskUsage
from app.schemas.visual import VisualPlanProposal, VisualRequirement

SPEECH = ProviderSpeechRequest(text="Hola amigos.", language="es-ES", voice_id="mock-es-ES-1", format="wav")
KEY = "sk-prod-test-0123456789abcdef"


def real(**over) -> ProviderSettings:
    values = dict(teaching_agent_mode="production", teaching_agent_offline=False, llm_routes={},
                  llm_provider="openai", llm_model="gpt-test", openai_api_key=KEY, tts_provider="openai",
                  image_provider="openai", search_provider="tavily", search_api_key="tvly-0123456789abcdef")
    values.update(over)
    return ProviderSettings(**values)


def record(capability: str, operation: str, *, status: str = "ok", provider: str = "openai", **kw):
    return ProviderRequestRecord(request_id=f"preq_{len(kw)}", provider=provider, capability=capability,
                                 operation=operation, status=status, **kw)


# --- mode and configuration ------------------------------------------------------------------------------------


def test_mode_is_explicit_and_never_follows_credentials() -> None:
    settings = ProviderSettings(teaching_agent_offline=False, llm_routes={}, openai_api_key=KEY)
    assert settings.mode == "offline" and settings.offline  # a key alone never switches to production
    assert real().mode == "production" and not real().offline
    forced = real(teaching_agent_offline=True)
    assert forced.mode == "offline"
    assert any("contradicts TEACHING_AGENT_MODE=production" in p for p in forced.problems())


def test_mode_and_budgets_load_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("TEACHING_AGENT_MODE", "production")
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)
    monkeypatch.setenv("MAX_LLM_REQUESTS", "12")
    monkeypatch.setenv("MAX_COST_USD", "1.5")
    monkeypatch.setenv("MAX_GENERATED_IMAGES", "")  # empty means unset: the default applies
    assert ProviderSettings(llm_routes={}).mode == "production"
    budget = ProductionSettings().budget()
    assert (budget.max_llm_requests, budget.max_cost_usd, budget.max_generated_images) == (12, 1.5, 4)
    monkeypatch.setenv("TEACHING_AGENT_MODE", "staging")
    with pytest.raises(ValueError):
        ProviderSettings(llm_routes={})


def test_readiness_names_each_missing_capability_and_never_a_secret() -> None:
    offline = ProviderSettings(teaching_agent_offline=False, llm_routes={}, openai_api_key=KEY)
    problems, _ = production_readiness(offline, set(PROVIDER_CAPABILITIES) - {Capability.IMAGE_SEARCH})
    text = "\n".join(problems)
    assert "set TEACHING_AGENT_MODE=production" in text
    for var in ("LLM_PROVIDER", "TTS_PROVIDER", "IMAGE_PROVIDER", "SEARCH_PROVIDER"):
        assert var in text, var  # "set VAR", or "VAR='mock' is a mock" where the environment chose a mock
    assert KEY not in text

    assert production_readiness(real(), {Capability.LLM, Capability.TTS, Capability.SEARCH, Capability.IMAGE}) == ([], [])
    missing_key, _ = production_readiness(real(search_api_key=None), {Capability.SEARCH})
    assert any("SEARCH_API_KEY" in p for p in missing_key)
    mock_tts, _ = production_readiness(real(tts_provider="mock"), {Capability.TTS})
    assert mock_tts == ["TTS_PROVIDER='mock' is a mock: production mode needs a real tts provider (one of ['openai'])"]


def test_required_capabilities_follow_the_workflow_and_the_image_budget() -> None:
    default = ProductionSettings().budget()
    assert required_capabilities(PROVIDER_CAPABILITIES, default) == {
        Capability.LLM, Capability.SEARCH, Capability.IMAGE, Capability.TTS}  # MAX_SEARCHED_IMAGES=0
    no_images = default.model_copy(update={"max_generated_images": 0, "max_searched_images": 2})
    assert required_capabilities(PROVIDER_CAPABILITIES, no_images) == {
        Capability.LLM, Capability.SEARCH, Capability.IMAGE_SEARCH, Capability.TTS}
    _, warnings = production_readiness(real(), {Capability.IMAGE_SEARCH})
    assert any("no real image search adapter" in w for w in warnings)


def test_budget_is_read_back_from_task_metadata() -> None:
    budget = ProductionSettings(max_llm_requests=5).budget()
    assert TaskBudget.of({"budget": budget.model_dump(mode="json")}) == budget
    assert TaskBudget.of({}) is None


# --- budgets ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget, capability, operation, usage, units, limit", [
    (TaskBudget(max_llm_requests=2), Capability.LLM, "generate", TaskUsage(llm_requests=2), None, "MAX_LLM_REQUESTS"),
    (TaskBudget(max_llm_tokens=100), Capability.LLM, "generate", TaskUsage(input_tokens=60, output_tokens=40), None,
     "MAX_LLM_TOKENS"),
    (TaskBudget(max_search_requests=1), Capability.SEARCH, "search", TaskUsage(search_requests=1), None,
     "MAX_SEARCH_REQUESTS"),
    (TaskBudget(max_generated_images=0), Capability.IMAGE, "generate", TaskUsage(), None, "MAX_GENERATED_IMAGES"),
    (TaskBudget(max_tts_characters=20), Capability.TTS, "synthesize", TaskUsage(tts_characters=15),
     {"characters": 6}, "MAX_TTS_CHARACTERS"),
    (TaskBudget(max_tts_seconds=10), Capability.TTS, "synthesize", TaskUsage(tts_seconds=10), {"characters": 1},
     "MAX_TTS_SECONDS"),
    (TaskBudget(max_cost_usd=1), Capability.LLM, "generate", TaskUsage(estimated_cost_usd=1.0), None, "MAX_COST_USD"),
])
def test_admit_refuses_a_request_that_would_exceed_the_budget(budget, capability, operation, usage, units,
                                                               limit) -> None:
    with pytest.raises(BudgetExceededError) as err:
        admit(budget, usage, capability, operation, units)
    assert err.value.limit == limit and limit in str(err.value) and "resume" in str(err.value)


def test_admit_allows_requests_within_the_budget_and_unbilled_operations() -> None:
    budget = TaskBudget(max_llm_requests=2, max_tts_characters=20, max_generated_images=0)
    admit(budget, TaskUsage(llm_requests=1), Capability.LLM, "generate")
    admit(budget, TaskUsage(tts_characters=14), Capability.TTS, "synthesize", {"characters": 6})
    admit(budget, TaskUsage(), Capability.TTS, "voices")  # listing voices is not billed
    admit(budget, TaskUsage(), Capability.IMAGE_SEARCH, "fetch")
    admit(TaskBudget(), TaskUsage(llm_requests=10_000), Capability.LLM, "generate")  # no limit set


def test_check_after_catches_limits_only_known_from_the_response() -> None:
    check_after(TaskBudget(max_llm_tokens=100), TaskUsage(input_tokens=100))
    with pytest.raises(BudgetExceededError, match="MAX_LLM_TOKENS"):
        check_after(TaskBudget(max_llm_tokens=100), TaskUsage(input_tokens=101))
    with pytest.raises(BudgetExceededError, match="MAX_COST_USD"):
        check_after(TaskBudget(max_cost_usd=0.5), TaskUsage(estimated_cost_usd=0.51))


def test_task_usage_counts_attempts_and_never_invents_prices() -> None:
    records = [
        record("llm", "generate", model="gpt-test", input_tokens=100, output_tokens=20),
        record("llm", "generate", model="priced", input_tokens=10, output_tokens=5, estimated_cost_usd=0.01),
        record("llm", "generate", status="failed", error_type="ProviderUnavailable"),
        record("search", "search", provider="tavily", actual_cost_usd=0.008, result_count=5),
        record("image", "generate", image_count=1, estimated_cost_usd=0.04),
        record("tts", "synthesize", characters=120, audio_seconds=7.5),
        record("tts", "voices"),
        record("llm", "generate", provider="mock", model="mock-large", input_tokens=1, output_tokens=1),
    ]
    usage = TaskUsage.from_records(records, video_render_seconds=3.25)
    assert (usage.llm_requests, usage.failed_requests, usage.search_requests, usage.image_generations,
            usage.tts_requests) == (4, 1, 1, 1, 1)
    assert (usage.input_tokens, usage.output_tokens, usage.llm_tokens) == (111, 26, 137)
    assert (usage.tts_characters, usage.tts_seconds, usage.video_render_seconds) == (120, 7.5, 3.25)
    assert usage.estimated_cost_usd == pytest.approx(0.058) and usage.actual_cost_usd == pytest.approx(0.008)
    assert usage.unpriced == ["llm:openai/gpt-test", "tts:openai"] and not usage.cost_complete


class CountingTTS(MockTTSProvider):
    def __init__(self, failures: list[Exception] | None = None) -> None:
        super().__init__()
        self.failures = list(failures or [])
        self.attempts = 0

    async def synthesize(self, request):
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return await super().synthesize(request)


async def no_sleep(_: float) -> None:
    return None


def ledger_scope(budget: TaskBudget) -> tuple[ExecutionScope, UsageLedger]:
    ledger = UsageLedger(CostSummary(), budget=budget)
    return ExecutionScope(events=EventBus(), usage=ledger, task_id="t1", node_id="synthesize_audio"), ledger


async def test_budget_stop_is_never_retried_and_never_falls_back() -> None:
    first, second = CountingTTS(), CountingTTS()
    second.name = "second"
    invoker = ProviderInvoker(policies={Capability.TTS: ProviderPolicy(max_attempts=3)}, sleep=no_sleep)
    managed = ManagedTTSProvider([first, second], invoker)
    sc, ledger = ledger_scope(TaskBudget(max_tts_characters=len(SPEECH.text) + 5))
    with active_scope(sc):
        await managed.synthesize(SPEECH)
        with pytest.raises(BudgetExceededError, match="MAX_TTS_CHARACTERS"):
            await managed.synthesize(SPEECH)
    assert (first.attempts, second.attempts) == (1, 0)  # the refused request was never sent anywhere
    [sent] = ledger.summary.provider_requests
    assert (sent.status, sent.characters, sent.node_id) == ("ok", len(SPEECH.text), "synthesize_audio")


async def test_failed_attempts_are_recorded_and_count_as_requests() -> None:
    tts = CountingTTS(failures=[ProviderUnavailable("503")])
    invoker = ProviderInvoker(policies={Capability.TTS: ProviderPolicy(max_attempts=3)}, sleep=no_sleep)
    sc, ledger = ledger_scope(TaskBudget())
    with active_scope(sc):
        await ManagedTTSProvider([tts], invoker).synthesize(SPEECH)
    failed, ok = ledger.summary.provider_requests
    assert (failed.status, failed.error_type, failed.attempt) == ("failed", "ProviderUnavailable", 1)
    assert (ok.status, ok.attempt) == ("ok", 2) and failed.request_id != ok.request_id
    assert ledger.usage().tts_requests == 2 and ledger.usage().tts_characters == len(SPEECH.text)


# --- visual plan image budget ------------------------------------------------------------------------------------


def visual(visual_id: str, source: str, fallback: str | None = None) -> VisualRequirement:
    """A visual whose `fallback` source is allowed by giving it a query or prompt for that source too."""
    extra = {"search_query": "rising sea levels"} if "search" in (source, fallback) else {}
    if "generate" in (source, fallback):
        extra["generation_prompt"] = "A diagram of the greenhouse effect"
    return VisualRequirement.model_validate(dict(
        visual_id=visual_id, purpose="Explain", lesson_section_id="s1", concept="Greenhouse effect",
        description="Diagram", visual_type="diagram", preferred_source=source, aspect_ratio="16:9", required=False, attribution_required=source == "search", **extra))


@pytest.mark.parametrize("requirements, budget, message", [
    ([visual("v1", "generate"), visual("v2", "generate")], {"max_generated_images": 1}, "budget allows 1"),
    ([visual("v1", "search")], {"max_searched_images": 0}, "may search"),
    ([visual("v1", "generate", "search")], {"max_searched_images": 0}, "fallbacks included"),
])
def test_visual_plan_over_the_image_budget_is_rejected_for_revision(requirements, budget, message) -> None:
    source = VisualPlanningInput.model_construct(lesson=SimpleNamespace(sections=[SimpleNamespace(section_id="s1")]),
                                                 max_visuals=4, **{"max_generated_images": None,
                                                                   "max_searched_images": None, **budget})
    with pytest.raises(OutputRejected, match=message):
        VisualAgent.check(None, VisualPlanProposal(requirements=requirements, rationale="r"), source)
    within = {k: v + len(requirements) for k, v in budget.items()}
    VisualAgent.check(None, VisualPlanProposal(requirements=requirements, rationale="r"),
                      source.model_copy(update=within))


# --- failure classification ------------------------------------------------------------------------------------


def chained(outer: Exception, inner: Exception) -> Exception:
    try:
        try:
            raise inner
        except Exception as exc:
            raise outer from exc
    except Exception as exc:
        return exc


@pytest.mark.parametrize("exc, node, expected", [
    (chained(RuntimeError("agent failed"), BudgetExceededError("b", limit="MAX_LLM_REQUESTS", used=3, maximum=2)),
     "teach_review", ("BudgetExceededError", "TeachingError")),
    (chained(RuntimeError("x"), ProviderAuthenticationError("401")), "synthesize_audio",
     ("ProviderError", "AudioError")),
    (ConfigError("bad"), "compose_video", ("ConfigurationError", "VideoError")),
    (ResearchRequired("research is mandatory and failed"), "research_policy", ("ResearchError", "ResearchError")),
    (ReviewRejected("rejected"), "teach_review", ("ReviewError", "ReviewError")),
    (RuntimeError("bad deck"), "render_presentation", ("PresentationError", "PresentationError")),
    (RuntimeError("bad plan"), "visual", ("VisualError", "VisualError")),
    (RuntimeError("no concepts"), "diagnose_1", ("PlanningError", "PlanningError")),
    (RuntimeError("?"), "something_else", ("WorkflowError", "WorkflowError")),
])
def test_failures_are_classified_by_cause_then_stage(exc, node, expected) -> None:
    category, stage = classify(exc, node)
    assert (category.value, stage.value) == expected


async def test_every_lesson_node_has_a_stage(container) -> None:
    from app.schemas.production import ProductionTask
    plan = await container.production.plan(ProductionTask(level="B1", topic="Climate change", language="es",
                                                           learner_id="l1"))
    assert plan.stages
    for node_id in plan.stages:
        assert stage_of(node_id) != FailureCategory.WORKFLOW, node_id


# --- production logs -------------------------------------------------------------------------------------------


def test_production_log_lines_are_structured_and_never_carry_secrets() -> None:
    register_secret(KEY)
    bus = EventBus()
    stream = io.StringIO()
    bus.subscribe(ProductionLogWriter(stream))
    bus.emit("provider.request_failed", task_id="t1", node_id="plan", agent_id="curriculum_planner",
             provider="openai", capability="llm", request_id="preq_1", latency_ms=12.5, error_type="ProviderError",
             error=f"401 Authorization: Bearer {KEY} rejected", headers={"Authorization": f"Bearer {KEY}"},
             prompt="the full prompt text")
    bus.emit("provider.request_completed", task_id="t1", node_id="plan", provider="openai", request_id="preq_2",
             latency_ms=40.0, usage={"input_tokens": 3})
    failed, ok = (json.loads(line) for line in stream.getvalue().splitlines())
    assert {"ts", "task_id", "node", "provider", "request_id", "duration_ms", "status"} <= set(failed)
    assert (failed["status"], failed["duration_ms"], ok["status"]) == ("failed", 12.5, "ok")
    text = stream.getvalue()
    assert KEY not in text and "Authorization" not in json.dumps(ok) and "full prompt" not in text
    assert "headers" not in failed and "usage" not in ok  # only the documented fields, never payloads
    assert production_record(bus.emit("task.created", task_id="t2"))["event"] == "task.created"
