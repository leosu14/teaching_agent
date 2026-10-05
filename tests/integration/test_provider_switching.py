"""Acceptance: the same agents and workflows run on mocks offline, and on configured real providers.

The real-provider run uses the production adapters (OpenAI-compatible chat, OpenAI speech and images, Tavily
search) end to end, over in-process fakes of the vendor HTTP APIs: request building, auth, request ids, response
parsing, the invoker and the managed providers all run for real; only the network is replaced.
"""

from __future__ import annotations

import pytest

from app.config.providers import ProviderSettings
from app.config.routing import ConfigError
from app.config.settings import Settings
from app.providers.image.openai import OpenAIImageGenerationProvider
from app.providers.llm.anthropic import AnthropicLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.search.tavily import TavilySearchProvider
from app.providers.tts.openai import OpenAITTSProvider
from app.schemas.artifact import ArtifactType
from app.schemas.providers import Capability
from app.schemas.task import TaskStatus
from app.services.container import build_container
from tests.conftest import FIXTURES, run_lesson
from tests.fake_vendors import FAKE_KEY, FakeAnthropic, FakeChatModel, FakeOpenAI, FakeTavily, http_client


def real_settings(tmp_path, **overrides) -> Settings:
    values = dict(teaching_agent_mode="production", teaching_agent_offline=False, llm_routes={}, llm_provider="openai", llm_model="gpt-test",
                  openai_api_key=FAKE_KEY, tts_provider="openai", image_provider="openai", search_provider="tavily",
                  search_api_key="tvly-test-0123456789abcdef", tts_languages="es-ES,en-US")
    values.update(overrides)
    return Settings(data_dir=tmp_path / "data", log_json=False, providers=ProviderSettings(**values))


def test_default_container_selects_mocks_offline(container) -> None:
    providers = container.providers
    assert providers.registry.offline is True
    for selection in providers.selector.describe():
        assert selection.provider == "mock", selection
    for cap, ids in providers.registry.capabilities().items():
        assert all(not providers.registry.get(cap, pid).requires_network for pid in ids)


async def test_mock_pipeline_emits_provider_events_with_unique_request_ids(container) -> None:
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    events = container.task_service.events(task.task_id)
    completed = [e for e in events if e.type == "provider.request_completed"]
    capabilities = {e.data["capability"] for e in completed}
    assert {"llm", "search", "tts", "image", "image_search"} <= capabilities
    ids = [e.data["request_id"] for e in completed]
    assert len(ids) == len(set(ids)) and all(i.startswith("preq_") for i in ids)
    assert all(e.task_id == task.task_id for e in completed)  # attributed to the task through the scope
    assert all(e.data["usage"]["request_id"] == e.data["request_id"] for e in completed)
    started = [e for e in events if e.type == "provider.request_started"]
    assert len(started) == len(completed)


def test_real_configuration_selects_production_adapters(tmp_path) -> None:
    c = build_container(real_settings(tmp_path))
    try:
        registry = c.providers.registry
        assert isinstance(registry.get(Capability.LLM, "openai"), OpenAICompatibleLLMProvider)
        assert isinstance(registry.get(Capability.TTS, "openai"), OpenAITTSProvider)
        assert isinstance(registry.get(Capability.IMAGE, "openai"), OpenAIImageGenerationProvider)
        assert isinstance(registry.get(Capability.SEARCH, "tavily"), TavilySearchProvider)
        assert registry.ids(Capability.IMAGE_SEARCH) == ["mock"]
        assert {s.capability: s.provider for s in c.providers.selector.describe()} == {
            Capability.LLM: "openai", Capability.TTS: "openai", Capability.IMAGE: "openai",
            Capability.IMAGE_SEARCH: "mock", Capability.SEARCH: "tavily", Capability.VIDEO_GENERATION: "mock"}
        assert all(t.provider == "openai" and t.model == "gpt-test" for t in c.router.targets(c.router.tier_for(
            "teacher", c.agents.get("teacher").spec.tier)))
        assert "gpt-test" in c.router.config.unpriced_models  # no price configured: cost unknown, not invented
    finally:
        c.close()


def test_real_configuration_is_rejected_in_offline_mode(tmp_path) -> None:
    with pytest.raises(ConfigError) as err:
        build_container(real_settings(tmp_path, teaching_agent_offline=True))
    message = str(err.value)
    assert "TEACHING_AGENT_OFFLINE=true" in message and "LLM_PROVIDER='openai'" in message
    assert FAKE_KEY not in message


def test_missing_credentials_are_reported_by_variable_name(tmp_path) -> None:
    with pytest.raises(ConfigError) as err:
        build_container(real_settings(tmp_path, openai_api_key=None, search_api_key=None))
    message = str(err.value)
    assert "set OPENAI_API_KEY" in message and "set TTS_API_KEY or OPENAI_API_KEY" in message
    assert "set SEARCH_API_KEY or TAVILY_API_KEY" in message


async def test_whole_pipeline_runs_on_real_adapters_over_fake_vendor_apis(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)
    chat = FakeChatModel(default_responders())
    openai, tavily = FakeOpenAI(chat), FakeTavily(FIXTURES / "web_corpus.json")
    base = "https://api.openai.example/v1"
    llm = OpenAICompatibleLLMProvider(http_client("openai", base, openai, request_id_header="X-Client-Request-Id"))
    tts = OpenAITTSProvider(http_client("openai", base, openai), languages=["es-ES", "en-US"])
    images = OpenAIImageGenerationProvider(http_client("openai", base, openai))
    search = TavilySearchProvider(http_client("tavily", "https://api.tavily.example", tavily))
    c = build_container(real_settings(tmp_path), llm_providers={"openai": llm}, tts_provider=tts,
                        image_generation_provider=images, search_provider=search)
    chat.prompts.update({c.agents.get(a.id).system_prompt: a.id for a in c.agents.describe()})
    try:
        task = await run_lesson(c)
        assert task.status == TaskStatus.COMPLETED, task.errors
        arts = c.task_service.artifacts(task.task_id)
        types = {a.type for a in arts}
        assert {ArtifactType.LESSON, ArtifactType.PRESENTATION, ArtifactType.VIDEO} <= types
        audio = [a for a in arts if a.type == ArtifactType.AUDIO_ASSET]
        assert audio and all(a.metadata["provider"] == "openai" for a in audio)

        events = c.task_service.events(task.task_id)
        completed = [e for e in events if e.type == "provider.request_completed"]
        by_provider = {(e.data["capability"], e.data["provider"]) for e in completed}
        assert {("llm", "openai"), ("tts", "openai"), ("search", "tavily")} <= by_provider
        llm_usage = [e.data["usage"] for e in completed if e.data["capability"] == "llm"]
        assert all(u["vendor_request_id"].startswith("vendor-") for u in llm_usage)
        assert all("estimated_cost" not in u for u in llm_usage)  # unpriced model: no invented cost

        chat_requests = [r for r in openai.requests if r.url.path == "/v1/chat/completions"]
        ids = [r.headers["X-Client-Request-Id"] for r in chat_requests]
        assert len(ids) == len(set(ids)) and set(ids) <= {u["request_id"] for u in llm_usage}
        assert all(r.headers["Authorization"] == f"Bearer {FAKE_KEY}" for r in chat_requests)
        assert any("response_format" in b for b in openai.bodies("/v1/chat/completions"))
        assert all(b["include_raw_content"] for b in tavily.bodies())
        assert FAKE_KEY not in str([e.model_dump() for e in events])
    finally:
        c.close()


async def test_anthropic_adapter_serves_an_agent_route(tmp_path, monkeypatch) -> None:
    """A per-agent route (the reviewer on another vendor) without touching the agent."""
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)
    chat = FakeChatModel(default_responders())
    fake = FakeAnthropic(chat)
    anthropic = AnthropicLLMProvider(http_client("anthropic", "https://api.anthropic.example", fake,
                                                 headers={"x-api-key": FAKE_KEY}, request_id_header=""))
    settings = Settings(data_dir=tmp_path / "data", log_json=False, providers=ProviderSettings(
        teaching_agent_mode="production", teaching_agent_offline=False, anthropic_api_key=FAKE_KEY,
        llm_routes={"content_reviewer": ("anthropic", "claude-test")}))
    from app.providers.llm.mock import MockLLMProvider
    c = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders()),
                                                 "anthropic": anthropic})
    chat.prompts.update({c.agents.get(a.id).system_prompt: a.id for a in c.agents.describe()})
    try:
        task = await run_lesson(c)
        assert task.status == TaskStatus.COMPLETED, task.errors
        calls = [e for e in c.task_service.events(task.task_id) if e.type == "llm.call"]
        assert {e.data["provider"] for e in calls if e.agent_id == "content_reviewer"} == {"anthropic"}
        assert {e.data["provider"] for e in calls if e.agent_id != "content_reviewer"} == {"mock"}
        assert fake.requests and all(r.headers["x-api-key"] == FAKE_KEY for r in fake.requests)
        assert all("output_config" in b or "system" in b for b in fake.bodies())
    finally:
        c.close()


def test_the_semantic_grader_uses_the_configured_llm_only_in_production(tmp_path, container) -> None:
    """No new provider configuration: the grader routes like every agent, to mocks offline (the default) and to the
    configured real provider only under TEACHING_AGENT_MODE=production."""
    offline = container.router.targets(container.router.tier_for(
        "semantic_grader", container.agents.get("semantic_grader").spec.tier))
    assert offline and all(t.provider == "mock" for t in offline)
    c = build_container(real_settings(tmp_path))
    try:
        targets = c.router.targets(c.router.tier_for("semantic_grader", c.agents.get("semantic_grader").spec.tier))
        assert targets and all(t.provider == "openai" and t.model == "gpt-test" for t in targets)
    finally:
        c.close()
