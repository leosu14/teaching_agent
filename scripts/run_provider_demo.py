"""Demo of the production provider infrastructure.

    python scripts/run_provider_demo.py            # offline, mock providers only (the default)
    python scripts/run_provider_demo.py --smoke    # also one minimal real request per configured real provider

By default it forces offline mode and mock providers, whatever the environment says, and prints:
the provider configuration (credentials only as set/missing), every registered provider with its capabilities and
health, the selected provider per capability and per agent, the fallback configuration, then one call per
capability through the provider layer (structured LLM output, TTS, image generation, image search, web search)
with each call's request id and usage, a retry and a fallback on deliberately failing mock providers, the
provider.* events, and a check that secrets are redacted.

With --smoke and real providers configured (e.g. LLM_PROVIDER=anthropic LLM_MODEL=... ANTHROPIC_API_KEY=...), it
validates that configuration and sends each configured real provider one minimal request. Without real
configuration the smoke part is skipped, not failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.providers import MOCK, ProviderSettings  # noqa: E402
from app.config.routing import ConfigError  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.observability.redaction import redact_text, register_secret  # noqa: E402
from app.observability.scope import ExecutionScope, UsageLedger  # noqa: E402
from app.providers.core.errors import ProviderRateLimit, ProviderUnavailable  # noqa: E402
from app.providers.core.invoker import ProviderInvoker  # noqa: E402
from app.providers.image.base import ProviderImageRequest  # noqa: E402
from app.providers.image_search.base import ProviderImageSearchRequest  # noqa: E402
from app.providers.llm.mock import MockLLMProvider  # noqa: E402
from app.providers.llm.mock_responders import default_responders  # noqa: E402
from app.providers.llm.structured import StructuredLLM  # noqa: E402
from app.providers.managed import ManagedTTSProvider  # noqa: E402
from app.providers.search.base import ProviderSearchRequest  # noqa: E402
from app.providers.tts.base import ProviderSpeechRequest  # noqa: E402
from app.providers.tts.mock import MockTTSProvider  # noqa: E402
from app.schemas.common import ModelTier  # noqa: E402
from app.schemas.events import Event  # noqa: E402
from app.schemas.providers import Capability, ProviderPolicy  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402

DEMO_AGENT = "provider_demo"


class LessonHook(BaseModel):
    """The demo's structured output: a Pydantic model, validated after parsing like every agent output."""

    title: str = Field(min_length=1)
    hook: str = Field(min_length=1)
    concepts: list[str] = Field(min_length=1)


def demo_responder(request) -> dict:
    return {"title": "Talking about football in Spanish", "hook": "¿Qué te pareció el partido? Give your opinion like a fan.",
            "concepts": ["football vocabulary", "me parece que + opinion", "the preterite for past matches"]}


class FlakyTTS(MockTTSProvider):
    """A mock TTS whose first calls fail: rate limited twice (retried), or down for good (falls back)."""

    def __init__(self, name: str, failures: list[Exception]) -> None:
        super().__init__()
        self.name = name
        self._failures = list(failures)

    async def synthesize(self, request):
        if self._failures:
            raise self._failures.pop(0)
        return await super().synthesize(request)


def heading(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def mock_settings(data_dir: Path) -> Settings:
    """Offline, mock-only: explicit values win over the environment and .env."""
    providers = ProviderSettings(teaching_agent_offline=True, llm_provider=None, llm_model=None,
                                 llm_fallback_provider=None, llm_fallback_model=None, llm_routes={},
                                 tts_provider=MOCK, tts_fallback_provider=None, image_provider=MOCK,
                                 image_fallback_provider=None, image_search_provider=MOCK, search_provider=MOCK,
                                 search_fallback_provider=None)
    return Settings(data_dir=data_dir, log_json=False, providers=providers)


def show_usage(container: Container, since: int) -> int:
    log = list(container.providers.invoker.usage_log)
    for u in log[since:]:
        units = {k: v for k, v in u.model_dump(exclude_none=True).items()
                 if k in ("input_tokens", "output_tokens", "characters", "audio_seconds", "image_count",
                          "result_count")}
        cost = f"{u.actual_cost:.6f} {u.currency} actual" if u.actual_cost is not None else (
            f"{u.estimated_cost:.6f} {u.currency} estimated" if u.estimated_cost is not None else "cost unknown")
        print(f"  {u.capability.value:<12} {u.provider:<8} {u.operation:<10} request_id={u.request_id} "
              f"attempt={u.attempt} latency={u.latency_ms}ms {units} {cost}")
    return len(log)


async def describe(container: Container) -> None:
    p = container.settings.providers
    heading("1. Configuration (credentials are never printed, only set/missing)")
    print(json.dumps(p.describe(), indent=1))
    policies = p.policies()
    for cap in Capability:
        pol = policies[cap]
        print(f"  policy {cap.value:<12} timeout={pol.timeout_seconds}s attempts<={pol.max_attempts} "
              f"backoff={pol.backoff_seconds}s x{pol.backoff_multiplier} (max {pol.max_backoff_seconds}s) "
              f"rpm={pol.requests_per_minute or '-'} concurrency={pol.max_concurrency or '-'}")

    heading("2. Registered providers and capabilities")
    registry = container.providers.registry
    for cap, ids in registry.capabilities().items():
        for pid in ids:
            provider = registry.get(cap, pid)
            print(f"  {cap.value:<12} {pid:<8} network={provider.requires_network!s:<5} "
                  f"config={provider.configuration() or '{}'}")

    heading("3. Health")
    for status in await registry.check_health():
        print(f"  {status.capability.value:<12} {status.provider:<8} available={status.available} "
              f"latency={status.latency_ms}ms checked={status.checked}" + (f" error={status.error}"
                                                                           if status.error else ""))

    heading("4. Selected providers")
    selector = container.providers.selector
    for selection in selector.describe():
        print(f"  {selection.capability.value:<12} -> {selection.provider} (model: {selection.model or '-'}; "
              f"{selection.reason}; available: {selection.available})")
    for agent in container.agents.describe():
        tier = container.router.tier_for(agent.id, agent.tier)
        s = selector.select(Capability.LLM, agent_id=agent.id, tier=tier)
        print(f"  llm for {agent.id:<22} -> {s.provider}/{s.model} ({s.reason})")

    heading("5. Fallback configuration")
    for cap in Capability:
        chain = selector.configured_chain(cap) if cap != Capability.LLM else None
        if cap == Capability.LLM:
            chains = {t.value: [f"{x.provider}/{x.model}" for x in container.router.targets(t)] for t in ModelTier}
            print(f"  llm          tier chains {chains}")
        else:
            print(f"  {cap.value:<12} {' -> '.join(chain) if len(chain) > 1 else 'no fallback configured'}")


async def calls(container: Container) -> None:
    heading("6. One call per capability through the provider layer")
    scope = ExecutionScope(events=container.events, usage=UsageLedger(), task_id="provider-demo")
    seen = 0
    hook = await StructuredLLM(container.router).generate(
        LessonHook, agent_id=DEMO_AGENT, tier=ModelTier.CHEAP, system="You write lesson hooks.",
        prompt="Write a lesson hook for talking about football in Spanish. Respond with JSON only.", scope=scope)
    print(f"  structured LLM output ({type(hook).__name__}): {hook.model_dump()}")
    p = container.providers
    speech = await p.tts.synthesize(ProviderSpeechRequest(text="Hola. ¿Qué te pareció el partido?", language="es-ES",
                                                          voice_id="mock-es-ES-1", format="wav"))
    print(f"  tts: {speech.provider} {speech.format} {speech.duration:.3f}s {len(speech.content)} bytes")
    image = await p.image_generation.generate(ProviderImageRequest(prompt="A timeline diagram of a football match", width=640,
                                                                   height=360))
    print(f"  image generation: {image.provider} origin={image.origin} {image.width}x{image.height} "
          f"{len(image.content)} bytes")
    images = await p.image_search.search(ProviderImageSearchRequest(query="football fans", max_results=3))
    print(f"  image search: {len(images.hits)} hits, origins {sorted({h.origin for h in images.hits})}, "
          f"licences {sorted({h.license_name or '-' for h in images.hits})}")
    page = await p.search.search(ProviderSearchRequest(query="football vocabulary in Spanish", max_results=3))
    print(f"  web search: {page.provider} {len(page.hits)} hits: {[h.url for h in page.hits]}")
    print("\n  usage records:")
    seen = show_usage(container, seen)
    print(f"\n  task ledger: llm_calls={scope.usage.summary.llm_calls} tokens={scope.usage.summary.token_usage.model_dump()}"
          f" estimated cost=${scope.usage.summary.actual_cost_usd:.6f}")

    heading("7. Retry and fallback (deliberately failing mock providers)")
    first_event = len(recorded)
    invoker = ProviderInvoker(policies={Capability.TTS: ProviderPolicy(max_attempts=3, backoff_seconds=0.01)},
                              events=container.events, offline=True)
    request = ProviderSpeechRequest(text="Uno, dos, tres.", language="es-ES", voice_id="mock-es-ES-1", format="wav")
    retried = ManagedTTSProvider([FlakyTTS("flaky", [ProviderRateLimit("429 slow down", retry_after=0.01)] * 2)],
                                 invoker)
    speech = await retried.synthesize(request)
    print(f"  rate limited twice, retried, then served by {speech.provider}: {speech.duration:.3f}s")
    down = [ProviderUnavailable("503 service unavailable")] * 3
    chain = ManagedTTSProvider([FlakyTTS("primary", down), MockTTSProvider()], invoker)
    speech = await chain.synthesize(request)
    print(f"  primary unavailable after 3 attempts -> explicitly configured fallback served it: {speech.provider}")
    for e in [e for e in recorded[first_event:] if e.type.startswith("provider.")]:
        d = e.data
        print(f"  event {e.type:<28} provider={d.get('provider') or d.get('from_provider')} "
              f"attempt={d.get('attempt', '-')} {('error=' + d['error_type']) if 'error_type' in d else ''}"
              f"{(' -> ' + d['to_provider']) if 'to_provider' in d else ''}")

    heading("8. Security: secrets are redacted")
    fake = "sk-demo-" + "x" * 24
    register_secret(fake)
    print(f"  error text with a key  -> {redact_text(f'401 for key {fake} (Authorization: Bearer {fake})')}")
    registry = container.providers.registry
    network = [pid for cap, ids in registry.capabilities().items() for pid in ids
               if registry.get(cap, pid).requires_network]
    print(f"  offline mode: {container.settings.providers.offline}; network providers registered: {network}")


async def smoke() -> int:
    heading("9. Real provider smoke test (--smoke)")
    settings = Settings(log_json=False)
    p = settings.providers
    real = {cap: [pid for pid in p.chain(cap) if pid != MOCK] for cap in Capability if cap != Capability.LLM}
    real[Capability.LLM] = [r for r in {p.llm_provider, *(v[0] for v in (p.llm_routes or {}).values())}
                            if r and r != MOCK]
    if not any(real.values()):
        print("  skipped: no real provider is configured (set e.g. LLM_PROVIDER, LLM_MODEL and its API key)")
        return 0
    if p.offline:
        print("  skipped: TEACHING_AGENT_OFFLINE=true")
        return 0
    try:
        container = build_container(Settings(data_dir=Path(tempfile.mkdtemp(prefix="provider-smoke-")),
                                             log_json=False))
    except ConfigError as exc:
        print(f"  configuration is invalid:\n{exc}")
        return 1
    failures = 0
    try:
        scope = ExecutionScope(events=container.events, usage=UsageLedger(), task_id="provider-smoke")
        for status in await container.providers.registry.check_health():
            print(f"  health {status.capability.value:<12} {status.provider:<9} available={status.available}"
                  + (f" error={status.error}" if status.error else ""))
        checks = []
        if real[Capability.LLM]:
            checks.append(("llm", lambda: StructuredLLM(container.router).generate(
                LessonHook, agent_id="teacher", tier=ModelTier.CHEAP, system="You write lesson hooks.",
                prompt="Return JSON with a title, a one-sentence hook and two concepts for a lesson on fractions.",
                scope=scope, max_output_tokens=400, timeout_seconds=120)))
        if real[Capability.TTS]:
            voice = (await container.providers.tts.voices())[0]
            checks.append(("tts", lambda: container.providers.tts.synthesize(ProviderSpeechRequest(
                text="Hello.", language=voice.language, voice_id=voice.voice_id, format="wav"))))
        if real[Capability.IMAGE]:
            checks.append(("image", lambda: container.providers.image_generation.generate(ProviderImageRequest(
                prompt="A simple flat illustration of a fraction pie chart", width=512, height=512))))
        if real[Capability.SEARCH]:
            checks.append(("search", lambda: container.providers.search.search(ProviderSearchRequest(
                query="what is a fraction", max_results=2))))
        for name, check in checks:
            try:
                result = await check()
                print(f"  smoke {name:<7} OK: {str(result)[:150]}")
            except Exception as exc:  # noqa: BLE001 - report every failure, then the exit code says it
                failures += 1
                print(f"  smoke {name:<7} FAILED: {type(exc).__name__}: {redact_text(str(exc))[:300]}")
        show_usage(container, 0)
    finally:
        container.close()
    return 1 if failures else 0


recorded: list[Event] = []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database (default: a fresh temporary directory)")
    parser.add_argument("--smoke", action="store_true",
                        help="also send one minimal request to each configured real provider (needs credentials)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")

    with tempfile.TemporaryDirectory(prefix="teaching-agent-provider-demo-") as tmp:
        responders = {**default_responders(), DEMO_AGENT: demo_responder}
        container = build_container(mock_settings(args.data_dir or Path(tmp)),
                                    llm_providers={"mock": MockLLMProvider(responders)})
        container.events.subscribe(recorded.append)
        try:
            asyncio.run(describe(container))
            asyncio.run(calls(container))
        finally:
            container.close()
    print("\nmock provider demo: OK (offline, no network, no credentials)")
    if args.smoke:
        return asyncio.run(smoke())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
