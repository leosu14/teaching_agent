"""Possibly billed transport failures (PR19 review F1), through the real HTTP client and adapters.

A response cut off after its headers arrived, or a gateway/server error (502, 504, 500), may follow an execution the
vendor billed: each such attempt is recorded `unknown` with its estimate charged, retried as before. A connection that
was never established (DNS, refused) proves nothing was sent: `failed`, request count only. A 503/529 overload
refusal stays `failed` too.
"""

from __future__ import annotations

import httpx
import pytest

from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.providers.core.errors import ProviderUnavailable
from app.providers.core.invoker import ProviderInvoker
from app.providers.llm.base import LLMMessage, LLMRequest
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.managed import ManagedLLMProvider, ManagedVideoGenerationProvider
from app.providers.video_generation.minimax import MiniMaxVideoGenerationProvider
from app.schemas.common import TokenUsage
from app.schemas.generative_video import VideoGenerationRequest
from app.schemas.providers import Capability, ProviderPolicy
from app.schemas.spend import SpendState
from tests.fake_vendors import http_client

ATTEMPTS = 3


@pytest.fixture(autouse=True)
def online(monkeypatch):
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)


class CutOff(httpx.AsyncByteStream):
    """A 200 body that breaks after its first bytes (the connection drops mid-response)."""

    async def __aiter__(self):
        yield b'{"id": "chatcmpl-1", "choices": [{"message": {"content": "{\\"gre'
        raise httpx.ReadError("connection reset by peer")


def cut_off(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/json"}, stream=CutOff())


def status(code: int):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"error": {"message": "upstream"}})
    return handler


def unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno -2] Name or service not known", request=request)


def setup():
    policy = ProviderPolicy(max_attempts=ATTEMPTS, backoff_seconds=0)

    async def no_sleep(_s: float) -> None:
        return None

    inv = ProviderInvoker(policies={c: policy for c in Capability}, sleep=no_sleep)
    ledger = UsageLedger()
    return inv, ExecutionScope(events=EventBus(), usage=ledger, task_id="t1"), ledger


def entries(ledger: UsageLedger):
    return ledger.account.store.entries()


def counters(ledger: UsageLedger, metric: str):
    [key] = ledger.account.scope_keys
    t = ledger.account.store.totals([key]).get(key, {}).get(metric)
    return (t.reserved, t.settled) if t else (0, 0)


def price(_model: str, usage: TokenUsage) -> float:
    return (usage.input_tokens + usage.output_tokens) * 1e-6


async def llm_call(handler) -> UsageLedger:
    inv, scope, ledger = setup()
    llm = ManagedLLMProvider(OpenAICompatibleLLMProvider(http_client("openai", "https://api.openai.example/v1",
                                                                     handler)), inv, cost=price)
    request = LLMRequest(model="m", system="You are terse.", messages=[LLMMessage(role="user", content="Say hi")],
                         max_output_tokens=50, agent_id="greeter")
    with active_scope(scope), pytest.raises(ProviderUnavailable):
        await llm.generate(request)
    return ledger


@pytest.mark.parametrize("handler", [cut_off, status(502), status(504), status(500)],
                         ids=["truncated-200", "502", "504", "500"])
async def test_a_possibly_billed_llm_failure_is_unknown_and_charged_its_estimate(handler) -> None:
    ledger = await llm_call(handler)
    es = entries(ledger)
    assert len(es) == ATTEMPTS  # still retried as before
    assert all(e.state == SpendState.UNKNOWN and e.usage_source == "estimate" for e in es)
    estimate = es[0].estimate["llm_tokens"]
    assert estimate > 50 and all(e.charged["llm_tokens"] == estimate for e in es)
    assert counters(ledger, "llm_tokens") == (0, ATTEMPTS * estimate)
    # The cost is the configured price of the estimate, labelled as such: never presented as reported.
    assert all(e.cost_source == "estimate" and e.final_cost_microusd for e in es)
    assert counters(ledger, "cost_microusd")[1] > 0


@pytest.mark.parametrize("handler", [unreachable, status(503), status(529)], ids=["dns", "503", "529"])
async def test_a_request_the_vendor_never_processed_is_failed(handler) -> None:
    ledger = await llm_call(handler)
    es = entries(ledger)
    assert len(es) == ATTEMPTS
    assert all(e.state == SpendState.FAILED and e.charged == {"llm_requests": 1} for e in es)
    assert counters(ledger, "llm_tokens") == (0, 0) and counters(ledger, "llm_requests") == (0, ATTEMPTS)


@pytest.mark.parametrize("handler,state", [(cut_off, SpendState.UNKNOWN), (status(502), SpendState.UNKNOWN),
                                           (status(504), SpendState.UNKNOWN), (unreachable, SpendState.FAILED)],
                         ids=["truncated-200", "502", "504", "dns"])
async def test_video_submission_failures_are_classified_the_same_way(handler, state) -> None:
    inv, scope, ledger = setup()
    minimax = MiniMaxVideoGenerationProvider(http_client("minimax", "https://api.minimax.example/v1", handler))
    video = ManagedVideoGenerationProvider([minimax], inv, price_per_second=0.5)
    request = VideoGenerationRequest(prompt="Educational clip of evaporation.", duration=6.0, width=1920,
                                     height=1080, fps=24, seed=3)
    with active_scope(scope), pytest.raises(ProviderUnavailable):
        await video.submit(request, generation_key="k1")
    es = entries(ledger)
    assert es and all(e.state == state for e in es)
    if state == SpendState.UNKNOWN:  # the job may exist at the vendor: its seconds and cost are charged
        assert all(e.charged["video_ms"] == 6000 and e.charged["video_cost_microusd"] == 3_000_000 for e in es)
        assert counters(ledger, "video_ms") == (0, 6000 * len(es))
    else:
        assert counters(ledger, "video_ms") == (0, 0) and counters(ledger, "video_segments") == (0, len(es))
