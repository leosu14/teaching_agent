"""Provider infrastructure: registry, selector, configuration, redaction, offline mode, health, errors, retry,
timeout, rate limiting, fallback, usage, request ids and events."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.config.providers import ProviderSettings, apply_llm_overrides, parse_llm_role_routes
from app.config.routing import ConfigError, load_routing
from app.config.settings import REPO_ROOT, Settings
from app.observability.events import EventBus
from app.observability.logging import JsonFormatter, log_event
from app.observability.redaction import REDACTED, redact, redact_text, register_secret
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.providers.core.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderInvalidRequest,
    ProviderOfflineError,
    ProviderRateLimit,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
    is_retryable,
)
from app.providers.core.invoker import ProviderInvoker
from app.providers.core.ratelimit import RateLimiter
from app.providers.core.registry import ProviderNotFound, ProviderRegistry
from app.providers.core.selector import ProviderSelector
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.managed import ManagedLLMProvider, ManagedSearchProvider, ManagedTTSProvider
from app.providers.search.base import ProviderSearchRequest, SearchPage, SearchProvider, SearchUsage
from app.providers.tts.base import ProviderSpeechRequest
from app.providers.tts.mock import MockTTSProvider
from app.schemas.common import ModelTier, TokenUsage
from app.schemas.events import Event
from app.schemas.providers import Capability, ProviderPolicy, ProviderUsage
from tests.unit.helpers import routing, scope

SPEECH = ProviderSpeechRequest(text="Hola amigos.", language="es-ES", voice_id="mock-es-ES-1", format="wav")


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def invoker(policy: ProviderPolicy | None = None, *, offline: bool = False, events: EventBus | None = None,
            sleep: Sleeps | None = None) -> ProviderInvoker:
    policy = policy or ProviderPolicy(max_attempts=3, backoff_seconds=0.5, max_backoff_seconds=4)
    return ProviderInvoker(policies={c: policy for c in Capability}, events=events, offline=offline,
                           sleep=sleep or Sleeps())


class ScriptedTTS(MockTTSProvider):
    def __init__(self, name: str = "scripted", failures: list[Exception] | None = None, *,
                 network: bool = False) -> None:
        super().__init__()
        self.name = name
        self.failures = list(failures or [])
        self.attempts = 0
        self.requires_network = network

    async def synthesize(self, request):
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return await super().synthesize(request)


class NetworkSearch(SearchProvider):
    name = "netsearch"
    requires_network = True

    async def search(self, request):  # pragma: no cover - must never run offline
        raise AssertionError("network provider ran in offline mode")


def bus() -> tuple[EventBus, list[Event]]:
    events = EventBus()
    seen: list[Event] = []
    events.subscribe(seen.append)
    return events, seen


# --- Registry ------------------------------------------------------------------------------------------------


def test_registry_registration_lookup_capabilities_and_defaults() -> None:
    registry = ProviderRegistry()
    first, second = MockTTSProvider(), ScriptedTTS("backup")
    registry.register(first)
    registry.register(second)
    registry.register(MockImageGenerationProvider())
    assert registry.get(Capability.TTS, "mock") is first
    assert registry.capabilities() == {Capability.IMAGE: ["mock"], Capability.TTS: ["backup", "mock"]}
    assert registry.default(Capability.TTS) is first  # first registered
    registry.set_default(Capability.TTS, "backup")
    assert registry.default(Capability.TTS) is second
    with pytest.raises(ProviderNotFound, match="no 'tts' provider 'nope'"):
        registry.get(Capability.TTS, "nope")
    with pytest.raises(ConfigError, match="does not provide capability 'search'"):
        registry.register(MockTTSProvider(), capability=Capability.SEARCH)
    with pytest.raises(ConfigError, match="already registered"):
        registry.register(MockTTSProvider())  # a different instance under the same id
    with pytest.raises(ProviderNotFound):
        registry.default(Capability.LLM)


def test_registry_offline_rejects_network_providers() -> None:
    registry = ProviderRegistry(offline=True)
    registry.register(MockTTSProvider())
    with pytest.raises(ProviderOfflineError, match="only mock providers"):
        registry.register(NetworkSearch())


async def test_health_checks_are_structured_and_bounded() -> None:
    class Broken(ScriptedTTS):
        async def probe(self) -> str:
            raise ProviderAuthenticationError("HTTP 401: invalid x-api-key sk-abcdefghijklmnop")

    class Hanging(ScriptedTTS):
        async def probe(self) -> str:
            await asyncio.sleep(5)
            return "network"

    registry = ProviderRegistry()
    for provider in (MockTTSProvider(), Broken("broken"), Hanging("hanging")):
        registry.register(provider)
    statuses = {s.provider: s for s in await registry.check_health(timeout_seconds=0.05)}
    ok, broken, hanging = statuses["mock"], statuses["broken"], statuses["hanging"]
    assert ok.available and ok.capability == Capability.TTS and ok.latency_ms is not None and ok.error is None
    assert not broken.available and broken.error.startswith("ProviderAuthenticationError")
    assert "sk-abcdefghijklmnop" not in broken.error  # credential-shaped text is redacted
    assert not hanging.available and "timed out" in hanging.error
    assert registry.last_health(Capability.TTS, "broken") == broken

    offline = ProviderRegistry()
    offline.register(NetworkSearch())
    offline.offline = True
    [status] = await offline.check_health()
    assert not status.available and "offline" in status.error


# --- Selector ------------------------------------------------------------------------------------------------


async def test_selector_uses_configuration_and_skips_unavailable_only_with_a_fallback() -> None:
    registry = ProviderRegistry()
    primary, backup = ScriptedTTS("primary"), ScriptedTTS("backup")
    registry.register(primary)
    registry.register(backup)
    selector = ProviderSelector(registry, chains={Capability.TTS: ["primary", "backup"]},
                                models={Capability.TTS: "tts-model"})
    selection = selector.select(Capability.TTS)
    assert (selection.provider, selection.model, selection.fallbacks) == ("primary", "tts-model", ("backup",))
    assert selection.available is None  # never checked
    assert selector.chain(Capability.TTS) == [primary, backup]
    assert selector.select(Capability.TTS, provider_id="backup").provider == "backup"

    async def down() -> str:
        raise ProviderUnavailable("down")

    primary.probe = down
    await registry.check_health(Capability.TTS)
    selection = selector.select(Capability.TTS)
    assert selection.provider == "backup" and selection.skipped == ("primary",) and "skipped" in selection.reason

    single = ProviderSelector(registry, chains={Capability.TTS: ["primary"]})
    assert single.select(Capability.TTS).provider == "primary"  # no fallback configured: never silently switched
    with pytest.raises(ProviderNotFound):
        ProviderSelector(registry, chains={Capability.TTS: ["ghost"]}).select(Capability.TTS)


def test_selector_llm_follows_agent_routes_then_tiers() -> None:
    config = routing(("mock", "m1"), ("backup", "m2"))
    config.routes["content_reviewer"] = [config.tiers[ModelTier.CHEAP][0].model_copy(update={"model": "m-review"})]
    registry = ProviderRegistry()
    registry.register(MockLLMProvider({}))
    registry.register(MockLLMProvider({}, name="backup"))
    selector = ProviderSelector(registry, routing=config)
    tiered = selector.select(Capability.LLM, agent_id="teacher", tier=ModelTier.REASONING)
    assert (tiered.provider, tiered.model, tiered.fallbacks, tiered.fallback_models) == ("mock", "m1", ("backup",),
                                                                                        ("m2",))
    routed = selector.select(Capability.LLM, agent_id="content_reviewer")
    assert (routed.provider, routed.model, routed.reason) == ("mock", "m-review", "agent route")


# --- Configuration -------------------------------------------------------------------------------------------


def test_provider_settings_load_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value-123")
    monkeypatch.setenv("LLM_TEACHER_PROVIDER", "openai")
    monkeypatch.setenv("LLM_TEACHER_MODEL", "gpt-teacher")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-456")
    monkeypatch.setenv("LLM_REVIEWER_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_REVIEWER_MODEL", "claude-review")
    monkeypatch.setenv("TTS_PROVIDER", "openai")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("SEARCH_REQUESTS_PER_MINUTE", "30")
    monkeypatch.setenv("SEARCH_INCLUDE_DOMAINS", "wikipedia.org, britannica.com")
    monkeypatch.setenv("TEACHING_AGENT_OFFLINE", "false")
    p = ProviderSettings()
    assert (p.llm_provider, p.llm_model, p.tts_provider, p.offline) == ("anthropic", "claude-test", "openai", False)
    assert p.llm_routes == {"teacher": ("openai", "gpt-teacher"), "content_reviewer": ("anthropic", "claude-review")}
    assert p.credential(Capability.TTS, "openai") == ("sk-openai-secret-456", "OPENAI_API_KEY")
    assert p.policies()[Capability.TTS].timeout_seconds == 12.5
    assert p.policies()[Capability.SEARCH].requests_per_minute == 30
    assert p.include_domains() == ["wikipedia.org", "britannica.com"]
    assert p.problems() == []
    described = json.dumps(p.describe())
    assert "secret" not in described and "set (ANTHROPIC_API_KEY)" in described
    assert "sk-ant-secret-value-123" not in repr(p) and "sk-ant-secret-value-123" not in str(p.model_dump())


def test_role_routes_ignore_reserved_names_and_map_aliases() -> None:
    routes = parse_llm_role_routes({"LLM_FALLBACK_PROVIDER": "x", "LLM_FALLBACK_MODEL": "y",
                                    "LLM_SLIDE_PLANNER_PROVIDER": "openai", "LLM_SLIDE_PLANNER_MODEL": "m",
                                    "LLM_REVIEWER_MODEL": "only-model", "UNRELATED": "1"})
    assert routes == {"slide_planner": ("openai", "m"), "content_reviewer": (None, "only-model")}


def test_default_role_is_the_llm_provider() -> None:
    p = ProviderSettings(llm_routes={"default": ("openai", "gpt-x"), "teacher": ("anthropic", "c")})
    assert (p.llm_provider, p.llm_model, p.llm_routes) == ("openai", "gpt-x", {"teacher": ("anthropic", "c")})
    explicit = ProviderSettings(llm_provider="anthropic", llm_model="c", llm_routes={"default": ("openai", "gpt-x")})
    assert (explicit.llm_provider, explicit.llm_model) == ("anthropic", "c")


def test_configuration_problems_name_variables_and_never_secrets() -> None:
    p = ProviderSettings(llm_routes={}, llm_provider="openai", llm_fallback_provider="anthropic",
                         tts_provider="elevenlabs", search_provider="tavily", search_fallback_provider="tavily",
                         llm_input_price_per_mtok=1.0, teaching_agent_offline=True, anthropic_api_key="sk-ant-zzzzzzzz")
    problems = "\n".join(p.problems())
    for expected in ("LLM_PROVIDER='openai' needs a credential: set OPENAI_API_KEY",
                     "LLM_PROVIDER='openai' needs LLM_MODEL",
                     "LLM_FALLBACK_PROVIDER and LLM_FALLBACK_MODEL must be set together",
                     "TTS_PROVIDER='elevenlabs' is not a known tts provider",
                     "SEARCH_FALLBACK_PROVIDER repeats SEARCH_PROVIDER",
                     "TEACHING_AGENT_OFFLINE=true allows only mock providers",
                     "LLM_INPUT_PRICE_PER_MTOK and LLM_OUTPUT_PRICE_PER_MTOK must be set together"):
        assert expected in problems, expected
    assert "sk-ant-zzzzzzzz" not in problems
    with pytest.raises(ConfigError, match="provider configuration is invalid"):
        p.validate_startup()
    with pytest.raises(ConfigError, match="LLM_PROVIDER='openai' needs"):
        Settings(providers=p).validate_runtime()


def test_default_configuration_is_mock_and_valid() -> None:
    p = ProviderSettings(llm_routes={})
    assert all(p.primary(c) in (None, "mock") for c in Capability) and p.problems() == []
    assert {c: p.chain(c) for c in Capability if c != Capability.LLM} == {
        Capability.TTS: ["mock"], Capability.IMAGE: ["mock"], Capability.IMAGE_SEARCH: ["mock"],
        Capability.SEARCH: ["mock"]}


def test_llm_overrides_replace_tiers_add_routes_and_never_invent_prices() -> None:
    base = load_routing(REPO_ROOT / "config" / "routing.toml")
    p = ProviderSettings(llm_routes={"teacher": ("anthropic", "claude-teacher")}, llm_provider="openai",
                         llm_model="gpt-x", llm_fallback_provider="anthropic", llm_fallback_model="claude-y",
                         llm_temperature=0.2)
    config = apply_llm_overrides(base, p)
    for tier in ModelTier:
        assert [(t.provider, t.model) for t in config.tiers[tier]] == [("openai", "gpt-x"), ("anthropic", "claude-y")]
    assert [(t.provider, t.model) for t in config.targets_for("teacher", ModelTier.REASONING)] == [
        ("anthropic", "claude-teacher")]
    assert config.temperature == 0.2 and config.providers() == ["anthropic", "openai"]
    assert config.unpriced_models == ["claude-teacher", "claude-y", "gpt-x"]
    assert config.cost("gpt-x", TokenUsage(input_tokens=1000, output_tokens=1000)) is None
    config.validate_against({"openai", "anthropic"})  # unpriced env models are allowed, with unknown cost
    priced = apply_llm_overrides(base, p.model_copy(update={"llm_input_price_per_mtok": 2.0,
                                                            "llm_output_price_per_mtok": 8.0}))
    assert priced.cost("gpt-x", TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)) == pytest.approx(10.0)
    assert apply_llm_overrides(base, ProviderSettings(llm_routes={})).tiers == base.tiers  # unset: file decides


# --- Secret redaction ----------------------------------------------------------------------------------------


def test_redaction_of_keys_registered_secrets_and_credential_shapes() -> None:
    register_secret("custom-secret-value-xyz")
    register_secret("abc")  # too short to register safely: ignored
    data = {"api_key": "plain", "Authorization": "Bearer abcdefghijklmnop", "nested": [{"x-api-key": "k"}],
            "message": "failed with custom-secret-value-xyz and sk-proj-abcdefghijkl", "input_tokens": 12,
            "url": "https://x.example/?api_key=supersecret1&q=1", "note": "abc stays"}
    clean = redact(data)
    assert clean["api_key"] == REDACTED and clean["Authorization"] == REDACTED
    assert clean["nested"] == [{"x-api-key": REDACTED}]
    assert clean["message"] == f"failed with {REDACTED} and {REDACTED}"
    assert clean["input_tokens"] == 12 and clean["note"] == "abc stays"
    assert "supersecret1" not in clean["url"] and "q=1" in clean["url"]
    assert redact_text("Authorization: Bearer tvly-abcdefghijklmn") == f"Authorization: {REDACTED}"


def test_event_logs_are_redacted(caplog) -> None:
    register_secret("sk-live-0000111122223333")
    event = Event(event_id="e1", type="provider.request_failed",
                  data={"error": "401 sk-live-0000111122223333", "headers": {"authorization": "Bearer x"}})
    with caplog.at_level(logging.INFO, logger="teaching_agent.events"):
        log_event(event)
    formatted = JsonFormatter().format(caplog.records[-1])
    assert "sk-live-0000111122223333" not in formatted and REDACTED in formatted


# --- Errors and retry ----------------------------------------------------------------------------------------


def test_error_model_transience_and_retryability() -> None:
    assert ProviderError("x").transient and not ProviderError("x", transient=False).transient
    for cls in (ProviderTimeout, ProviderRateLimit, ProviderUnavailable):
        assert cls("x").transient and is_retryable(cls("x"))
    for cls in (ProviderAuthenticationError, ProviderInvalidRequest, ProviderResponseError, ProviderOfflineError):
        assert not cls("x").transient and not is_retryable(cls("x"))
    assert not is_retryable(ProviderError("generic"))  # only typed transient failures are retried here
    assert isinstance(ProviderTimeout("x"), TimeoutError)
    assert ProviderRateLimit("x", retry_after=3).retry_after == 3


@pytest.mark.parametrize("failure", [ProviderTimeout("slow"), ProviderRateLimit("429"), ProviderUnavailable("503"),
                                     ConnectionError("reset")])
async def test_transient_failures_are_retried_with_exponential_backoff(failure) -> None:
    sleeps = Sleeps()
    provider = ScriptedTTS(failures=[failure, failure])
    managed = ManagedTTSProvider([provider], invoker(sleep=sleeps))
    speech = await managed.synthesize(SPEECH)
    assert speech.provider == "scripted" and provider.attempts == 3
    assert sleeps.delays == [0.5, 1.0]


@pytest.mark.parametrize("failure", [ProviderAuthenticationError("401"), ProviderInvalidRequest("400"),
                                     ProviderResponseError("bad schema"), ProviderError("generic, left to callers")])
async def test_permanent_failures_are_not_retried(failure) -> None:
    sleeps = Sleeps()
    provider = ScriptedTTS(failures=[failure])
    with pytest.raises(type(failure)):
        await ManagedTTSProvider([provider], invoker(sleep=sleeps)).synthesize(SPEECH)
    assert provider.attempts == 1 and sleeps.delays == []


async def test_retries_are_bounded_and_honour_retry_after_up_to_the_cap() -> None:
    sleeps = Sleeps()
    provider = ScriptedTTS(failures=[ProviderRateLimit("429", retry_after=2.0), ProviderRateLimit("429", retry_after=99),
                                     ProviderRateLimit("429"), ProviderRateLimit("never reached")])
    with pytest.raises(ProviderRateLimit) as err:
        await ManagedTTSProvider([provider], invoker(sleep=sleeps)).synthesize(SPEECH)
    assert provider.attempts == 3 and sleeps.delays == [2.0, 4.0]  # retry-after honoured, capped at max_backoff
    assert err.value.request_id.startswith("preq_") and err.value.provider == "scripted"
    assert ProviderPolicy(max_attempts=3, timeout_seconds=10, backoff_seconds=1).deadline_seconds() == 33.0
    with pytest.raises(ValueError):
        ProviderPolicy(max_attempts=11)  # unbounded retry is not configurable


async def test_timeouts_become_typed_and_are_retried() -> None:
    class Slow(ScriptedTTS):
        async def synthesize(self, request):
            self.attempts += 1
            await asyncio.sleep(1)

    provider = Slow()
    policy = ProviderPolicy(timeout_seconds=0.02, max_attempts=2, backoff_seconds=0)
    with pytest.raises(ProviderTimeout, match="within 0.02s"):
        await ManagedTTSProvider([provider], invoker(policy)).synthesize(SPEECH)
    assert provider.attempts == 2


# --- Rate limiting -------------------------------------------------------------------------------------------


async def test_requests_per_minute_window_waits_and_reports() -> None:
    now = [0.0]
    waits: list[tuple[str, float | None]] = []

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    limiter = RateLimiter(requests_per_minute=2, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        async with limiter.slot(lambda reason, s: waits.append((reason, s))):
            now[0] += 1
    assert waits == [("requests_per_minute", 58.0)] and now[0] == 61.0


async def test_concurrency_limit_and_rate_limited_events() -> None:
    events, seen = bus()
    running, peak = [0], [0]

    class Busy(ScriptedTTS):
        async def synthesize(self, request):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            await asyncio.sleep(0.01)
            running[0] -= 1
            return await super().synthesize(request)

    managed = ManagedTTSProvider([Busy()], invoker(ProviderPolicy(max_concurrency=2), events=events))
    await asyncio.gather(*(managed.synthesize(SPEECH) for _ in range(5)))
    assert peak[0] == 2
    limited = [e for e in seen if e.type == "provider.rate_limited"]
    assert limited and all(e.data["source"] == "local" and e.data["reason"] == "concurrency" for e in limited)


# --- Offline mode --------------------------------------------------------------------------------------------


async def test_offline_mode_rejects_network_providers_before_any_call() -> None:
    provider = ScriptedTTS("cloud", network=True)
    with pytest.raises(ProviderOfflineError, match="TEACHING_AGENT_OFFLINE=true"):
        await ManagedTTSProvider([provider], invoker(offline=True)).synthesize(SPEECH)
    assert provider.attempts == 0
    speech = await ManagedTTSProvider([MockTTSProvider()], invoker(offline=True)).synthesize(SPEECH)
    assert speech.provider == "mock"  # mocks run offline


# --- Fallback ------------------------------------------------------------------------------------------------


async def test_fallback_only_when_configured_on_transient_failure_and_recorded() -> None:
    events, seen = bus()
    primary = ScriptedTTS("primary", failures=[ProviderUnavailable("503")] * 3)
    backup = ScriptedTTS("backup")
    speech = await ManagedTTSProvider([primary, backup], invoker(events=events)).synthesize(SPEECH)
    assert speech.provider == "backup" and primary.attempts == 3 and backup.attempts == 1
    [fallback] = [e for e in seen if e.type == "provider.fallback"]
    assert fallback.data["from_provider"] == "primary" and fallback.data["to_provider"] == "backup"
    assert fallback.data["error_type"] == "ProviderUnavailable" and fallback.data["request_id"].startswith("preq_")

    fatal = ScriptedTTS("primary", failures=[ProviderAuthenticationError("401")])
    other = ScriptedTTS("backup")
    with pytest.raises(ProviderAuthenticationError):  # a permanent failure is never masked by a fallback
        await ManagedTTSProvider([fatal, other], invoker()).synthesize(SPEECH)
    assert other.attempts == 0

    alone = ScriptedTTS("alone", failures=[ProviderUnavailable("503")] * 3)
    with pytest.raises(ProviderUnavailable):  # no fallback configured: the failure surfaces
        await ManagedTTSProvider([alone], invoker()).synthesize(SPEECH)


# --- Usage, request ids and events ---------------------------------------------------------------------------


async def test_llm_usage_cost_request_ids_and_events_in_the_task_scope() -> None:
    sc, seen = scope()
    inv = invoker()
    llm = ManagedLLMProvider(MockLLMProvider({"a": lambda r: {"ok": True}}), inv,
                             cost=lambda model, usage: 0.25 if model == "priced" else None)
    from app.providers.llm.base import LLMMessage, LLMRequest
    request = LLMRequest(model="priced", system="s", messages=[LLMMessage(role="user", content="hi")],
                         max_output_tokens=10, agent_id="a")
    with active_scope(sc):
        await llm.generate(request)
        await llm.generate(request.model_copy(update={"model": "unpriced"}))
    priced, unpriced = list(inv.usage_log)
    assert isinstance(priced, ProviderUsage) and priced.capability == Capability.LLM and priced.provider == "mock"
    assert priced.input_tokens > 0 and priced.output_tokens > 0 and priced.latency_ms is not None
    assert (priced.estimated_cost, priced.currency, priced.actual_cost) == (0.25, "USD", None)
    assert (unpriced.estimated_cost, unpriced.currency) == (None, None)  # unknown, never invented
    assert priced.request_id != unpriced.request_id
    types = [e.type for e in seen]
    assert types == ["provider.request_started", "provider.request_completed"] * 2
    assert all(e.task_id == "t1" for e in seen)
    completed = [e for e in seen if e.type == "provider.request_completed"]
    assert [e.data["usage"]["request_id"] for e in completed] == [priced.request_id, unpriced.request_id]
    assert "content" not in json.dumps([e.data for e in seen])  # no prompts or payloads in provider events


async def test_failed_events_carry_typed_error_without_secrets() -> None:
    register_secret("sk-failing-key-123456")
    events, seen = bus()
    provider = ScriptedTTS(failures=[ProviderAuthenticationError("401: key sk-failing-key-123456 rejected",
                                                                 status=401)])
    with pytest.raises(ProviderAuthenticationError):
        await ManagedTTSProvider([provider], invoker(events=events)).synthesize(SPEECH)
    [failed] = [e for e in seen if e.type == "provider.request_failed"]
    assert failed.data["error_type"] == "ProviderAuthenticationError" and failed.data["status"] == 401
    assert failed.data["will_retry"] is False and "sk-failing-key-123456" not in json.dumps(failed.data)


async def test_search_usage_and_actual_provider_after_fallback() -> None:
    class Down(SearchProvider):
        name = "down"

        async def search(self, request):
            raise ProviderUnavailable("503")

    class Up(SearchProvider):
        name = "up"

        async def search(self, request):
            return SearchPage(hits=[], usage=SearchUsage(requests=1, results=0, cost_usd=0.002))

    inv = invoker(ProviderPolicy(max_attempts=1))
    page = await ManagedSearchProvider([Down(), Up()], inv).search(ProviderSearchRequest(query="q", max_results=1))
    assert page.provider == "up"
    [usage] = list(inv.usage_log)
    assert (usage.provider, usage.actual_cost, usage.currency, usage.result_count) == ("up", 0.002, "USD", 0)


def test_usage_ledger_scope_is_restored_after_the_block() -> None:
    sc = ExecutionScope(events=EventBus(), usage=UsageLedger())
    from app.observability.scope import current_scope
    assert current_scope() is None
    with active_scope(sc):
        assert current_scope() is sc
    assert current_scope() is None
