"""The run-wide cap and complete pricing (MVP item 10, issue #52).

- TTS, image generation and search have configured prices: each paid attempt reserves its priced cost before it is
  sent and settles it as `priced`.
- TA_BUDGET_TOTAL_MAX_COST_USD is one scope (`total:all`) that every durable account charges, whatever the kind of
  work, the learner or the UTC day: a request that would take it past the cap is refused before dispatch, concurrent
  reservations cannot pass it together, every retry attempt is reserved on its own, and it never resets at midnight.
- An LLM request's cost is reserved at a strict upper bound (UTF-8 bytes of the input, plus the output limit), so the
  settled cost of an answer within its limits never exceeds its reservation; a reported overrun stops further work.
- With the cap on, a paid call outside it (no work scope, or an in-memory account) is refused (fail closed), and a
  configuration with an unpriced paid capability, a non-finite price or a price for no model does not start.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.config.providers import ProviderSettings
from app.config.routing import ConfigError, ModelPricing
from app.config.settings import Settings
from app.observability.budget import BudgetExceededError
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.observability.spend import (
    TOTAL_SCOPE,
    InMemorySpendStore,
    attempt_account,
    session_account,
    task_account,
)
from app.providers.core.errors import ProviderTimeout
from app.providers.core.invoker import ProviderInvoker
from app.providers.image.base import ProviderImageRequest
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.llm.base import LLMMessage, LLMRequest
from app.providers.managed import (
    ManagedImageGenerationProvider,
    ManagedSearchProvider,
    ManagedTTSProvider,
    input_token_bound,
    llm_estimate,
)
from app.providers.search.base import ProviderSearchRequest, SearchPage, SearchProvider, SearchUsage
from app.providers.tts.base import ProviderSpeechRequest
from app.providers.tts.mock import MockTTSProvider
from app.schemas.common import ProviderRequestRecord, TokenUsage, new_id
from app.schemas.events import Event, EventType
from app.schemas.providers import Capability, ProviderPolicy, ProviderUsage
from app.schemas.spend import Estimate, SpendState
from app.schemas.usage import BudgetDefaults, TaskBudget
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlSpendRepository

BEFORE = datetime(2026, 11, 15, 23, 59, 30, tzinfo=timezone.utc)
SPEECH = ProviderSpeechRequest(text="Hola amigos, ¿qué tal?", language="es-ES", voice_id="mock-es-ES-1",
                               format="wav")


class Clock:
    def __init__(self, now: datetime = BEFORE) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture(params=["sql", "memory"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemorySpendStore()
        return
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    yield SqlSpendRepository(sessions)
    dispose(sessions)


def capped(usd: float) -> BudgetDefaults:
    return BudgetDefaults(total=TaskBudget(max_cost_usd=usd))


def reserve(account, cost_usd: float, *, capability: Capability = Capability.LLM, operation: str = "generate"):
    n = new_id("preq")
    estimate = (Estimate(llm_tokens=10, cost_usd=cost_usd) if capability == Capability.LLM
                else Estimate(cost_usd=cost_usd))
    return account.reserve(entry_id=n, operation_id=f"pop_{n}", attempt=1, capability=capability,
                           operation=operation, provider="p", model="m", estimate=estimate, timeout_seconds=60)


def total(store) -> tuple[int, int]:
    t = store.totals([TOTAL_SCOPE]).get(TOTAL_SCOPE, {}).get("cost_microusd")
    return (t.reserved, t.settled) if t else (0, 0)


# --- one scope over all work, never reset -------------------------------------------------------------------------------


def test_every_kind_of_account_charges_the_same_total_scope_on_both_sides_of_midnight(store) -> None:
    clock = Clock()
    defaults = capped(1.0)
    accounts = [task_account(store, defaults, task_id="t1", learner_id="L1", budget=None, fence=None, clock=clock),
                session_account(store, defaults, session_id="s1", task_id="t1", learner_id="L2", clock=clock),
                attempt_account(store, defaults, attempt_id="a1", task_id=None, learner_id=None, clock=clock)]
    for account in accounts:
        assert TOTAL_SCOPE in account.scope_keys
    clock.now = BEFORE + timedelta(minutes=1)  # the next UTC day: the daily scopes would change, this one does not
    for account in accounts:
        assert TOTAL_SCOPE in account.scope_keys
    assert TOTAL_SCOPE not in task_account(store, BudgetDefaults(), task_id="t2", learner_id=None, budget=None,
                                           fence=None).scope_keys  # off unless configured


def test_the_cap_refuses_before_dispatch_across_work_learners_and_days(store) -> None:
    clock = Clock()
    defaults = capped(0.001)  # 1000 micro-USD for everything together
    task = task_account(store, defaults, task_id="t1", learner_id="L1", budget=None, fence=None, clock=clock)
    session = session_account(store, defaults, session_id="s1", task_id=None, learner_id="L2", clock=clock)
    reserve(task, 0.0006)
    clock.now = BEFORE + timedelta(minutes=1)  # after midnight UTC: the cap does not reset
    with pytest.raises(BudgetExceededError) as refused:
        reserve(session, 0.0006)
    assert refused.value.scope == "total" and refused.value.limit == "MAX_COST_USD"
    assert "never resets" in str(refused.value)
    assert total(store) == (600, 0)  # the refused attempt reserved nothing
    reserve(session, 0.0004)  # exactly up to the cap still fits
    assert total(store) == (1000, 0)


def test_a_reported_overrun_is_recorded_and_stops_further_work(store) -> None:
    defaults = capped(0.001)
    account = task_account(store, defaults, task_id="t1", learner_id=None, budget=None, fence=None)
    ledger = UsageLedger(account=account)
    reservation = ledger.reserve(entry_id="preq_1", operation_id="pop_1", attempt=1, capability=Capability.LLM,
                                 operation="generate", provider="p", model="m",
                                 estimate=Estimate(llm_tokens=10, cost_usd=0.0009), timeout_seconds=60)
    record = ProviderRequestRecord(request_id="preq_1", provider="p", capability="llm", operation="generate",
                                   input_tokens=5, output_tokens=5, estimated_cost_usd=0.0012)
    with pytest.raises(BudgetExceededError) as stopped:  # the money is spent: recorded, and nothing more is sent
        ledger.settle(reservation, record, SpendState.SETTLED)
    assert stopped.value.scope == "total"
    assert total(store) == (0, 1200)
    with pytest.raises(BudgetExceededError):
        reserve(account, 0.0)


def race_total(url: str) -> None:
    """Eight workers, each its own task and learner, race 40 reservations of 50 micro-USD for a 1000 micro-USD cap:
    exactly 20 are admitted, whatever the interleaving. Shared with the PostgreSQL suite."""
    sessions = create_db(url)
    store = SqlSpendRepository(sessions)
    key = new_id("run")
    defaults = capped(0.001)
    ok: list[int] = []
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        account = task_account(store, defaults, task_id=f"{key}-t{n}", learner_id=f"{key}-L{n}", budget=None,
                               fence=None)
        barrier.wait()
        for _ in range(5):
            try:
                reserve(account, 0.00005)
                ok.append(n)
            except BudgetExceededError:
                pass

    before = total(store)
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    after = total(store)
    assert len(ok) == max(0, (1000 - before[0] - before[1]) // 50) and after[0] + after[1] <= 1000, (ok, after)
    dispose(sessions)


def test_concurrent_reservations_never_pass_the_cap(tmp_path) -> None:
    race_total(f"sqlite:///{tmp_path / 'db.sqlite'}")


# --- retries, and the fail-closed guard of the invoker --------------------------------------------------------------


async def no_sleep(_seconds: float) -> None:
    return None


class Vendor:
    name = "vendor"
    requires_network = False


def invoker(*, attempts: int = 1, cap: float | None = 1.0) -> ProviderInvoker:
    policy = ProviderPolicy(max_attempts=attempts, timeout_seconds=5, backoff_seconds=0)
    return ProviderInvoker(policies={c: policy for c in Capability}, sleep=no_sleep,
                           required_scope=TOTAL_SCOPE if cap is not None else None, required_limit=cap or 0.0)


def usage(_result, request_id: str) -> ProviderUsage:
    return ProviderUsage(provider="vendor", capability=Capability.LLM, operation="generate", model="m",
                         request_id=request_id, input_tokens=1, output_tokens=1, estimated_cost=0.0001)


async def test_every_retry_attempt_is_reserved_and_the_cap_stops_the_retries(store) -> None:
    defaults = capped(0.0025)
    ledger = UsageLedger(account=task_account(store, defaults, task_id="t1", learner_id=None, budget=None,
                                              fence=None))
    scope = ExecutionScope(events=EventBus(), usage=ledger, task_id="t1")
    sent = 0

    async def times_out():
        nonlocal sent
        sent += 1
        raise ProviderTimeout("slow", provider="vendor")

    with active_scope(scope), pytest.raises(BudgetExceededError) as refused:
        await invoker(attempts=5).call(Vendor(), Capability.LLM, "generate", times_out, usage=usage, model="m",
                                       estimate=Estimate(llm_tokens=10, cost_usd=0.001))
    # each timed-out attempt may have been billed: charged its estimate; the third does not fit and is never sent
    assert sent == 2 and refused.value.scope == "total"
    assert total(store) == (0, 2000)


async def test_a_paid_call_outside_the_cap_is_refused_and_never_sent() -> None:
    sent = 0

    async def answer():
        nonlocal sent
        sent += 1
        return "ok"

    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(seen.append)
    local = ExecutionScope(events=bus, usage=UsageLedger(), task_id="t1")  # an in-memory account: not the cap's
    with active_scope(local), pytest.raises(BudgetExceededError, match="not be charged to the run-wide cap"):
        await invoker().call(Vendor(), Capability.LLM, "generate", answer, usage=usage, model="m",
                             estimate=Estimate(llm_tokens=10, cost_usd=0.0))
    with pytest.raises(BudgetExceededError):  # no work scope at all
        await invoker().call(Vendor(), Capability.TTS, "synthesize", answer, usage=usage)
    assert sent == 0
    assert [e.data["budget"] for e in seen if e.type == EventType.BUDGET_EXHAUSTED] == ["total"]
    # an operation that is not billed (listing voices) still runs, and without the cap nothing changes
    assert await invoker().call(Vendor(), Capability.TTS, "voices", answer, usage=usage) == "ok"
    assert await invoker(cap=None).call(Vendor(), Capability.LLM, "generate", answer, usage=usage) == "ok"


# --- prices of TTS, images and search ----------------------------------------------------------------------------------


class PagedSearch(SearchProvider):
    name = "pages"
    requires_network = False

    async def search(self, request):
        return SearchPage(hits=[], usage=SearchUsage(requests=1, results=0))


def priced_scope(store):
    account = task_account(store, capped(1.0), task_id="t1", learner_id=None, budget=None, fence=None)
    return ExecutionScope(events=EventBus(), usage=UsageLedger(account=account), task_id="t1")


def settled(store) -> list:
    return [e for e in store.entries(task_id="t1") if e.state == SpendState.SETTLED]


async def test_tts_image_and_search_reserve_and_settle_their_configured_price(store) -> None:
    inv = invoker()
    tts = ManagedTTSProvider([MockTTSProvider()], inv, price_per_mchar=15.0)
    image = ManagedImageGenerationProvider([MockImageGenerationProvider()], inv, price_per_image=0.04)
    search = ManagedSearchProvider([PagedSearch()], inv, price_per_request=0.008)
    with active_scope(priced_scope(store)):
        await tts.synthesize(SPEECH)
        await image.generate(ProviderImageRequest(prompt="a market in Madrid", width=512, height=512))
        await search.search(ProviderSearchRequest(query="saludos en español", max_results=3))
    by_capability = {e.capability: e for e in settled(store)}
    chars = len(SPEECH.text)
    assert by_capability["tts"].estimated_cost_microusd == round(chars * 15)  # 15 USD per million characters
    assert by_capability["image"].estimated_cost_microusd == 40_000
    assert by_capability["search"].estimated_cost_microusd == 8_000
    for e in by_capability.values():
        assert e.cost_source == "priced" and e.final_cost_microusd == e.estimated_cost_microusd
    assert total(store) == (0, round(chars * 15) + 48_000)


async def test_a_tiny_cap_refuses_a_priced_tts_request_before_the_provider_is_called(store) -> None:
    provider = MockTTSProvider()
    calls = 0
    original = provider.synthesize

    async def counted(request):
        nonlocal calls
        calls += 1
        return await original(request)

    provider.synthesize = counted
    account = task_account(store, capped(0.0001), task_id="t1", learner_id=None, budget=None, fence=None)
    scope = ExecutionScope(events=EventBus(), usage=UsageLedger(account=account), task_id="t1")
    with active_scope(scope), pytest.raises(BudgetExceededError) as refused:
        await ManagedTTSProvider([provider], invoker(), price_per_mchar=15.0).synthesize(SPEECH)  # $0.00033 > cap
    assert calls == 0 and refused.value.scope == "total" and total(store) == (0, 0)


# --- the LLM's upper bound ---------------------------------------------------------------------------------------------


def request(text: str, *, schema: dict | None = None, max_output: int = 200) -> LLMRequest:
    return LLMRequest(model="m", system="Eres un profesor de español.", agent_id="teacher",
                      messages=[LLMMessage(role="user", content=text)], response_schema=schema,
                      max_output_tokens=max_output)


@pytest.mark.parametrize("text", ["hello", "¿Qué tal? Ñandú, café, 日本語の文章, emoji 🙂🙂🙂", "x" * 5000])
def test_the_llm_cost_reservation_is_an_upper_bound_of_any_answer_within_its_limits(text: str) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    req = request(text, schema=schema)
    utf8 = len(req.system.encode()) + len(text.encode()) + len(str(schema).encode())
    assert input_token_bound(req) >= utf8  # a byte-level tokenizer never makes more tokens than bytes

    price = ModelPricing(input_per_mtok=2.5, output_per_mtok=10.0, cached_input_per_mtok=2.5)

    def cost(_model: str, u: TokenUsage) -> float:
        return (u.input_tokens * price.input_per_mtok + u.output_tokens * price.output_per_mtok) / 1_000_000

    reserved = llm_estimate(req, cost).cost_usd
    worst = cost("m", TokenUsage(input_tokens=utf8, output_tokens=req.max_output_tokens))
    assert reserved >= worst
    assert llm_estimate(req, None).cost_usd is None  # unpriced: no guess


# --- configuration fails closed ------------------------------------------------------------------------------------------


def production(**env: object) -> Settings:
    base = dict(teaching_agent_mode="production", teaching_agent_offline=False, llm_routes={}, llm_provider="openai", llm_model="gpt-x",
                openai_api_key="sk-test-0123456789", tts_provider="openai", image_provider="openai",
                search_provider="tavily", tavily_api_key="tvly-test-0123456789",
                llm_input_price_per_mtok=2.5, llm_output_price_per_mtok=10.0, tts_price_per_mchar=15.0,
                image_price_per_image=0.04, search_price_per_request=0.008)
    base.update(env)
    return Settings(providers=ProviderSettings(**base), budget_total_max_cost_usd=1.0)


def test_a_fully_priced_configuration_passes_and_the_cap_is_wired() -> None:
    settings = production()
    assert settings.pricing_gaps() == [] and settings.cost_cap_problems() == []
    settings.validate_runtime()
    assert settings.budget_defaults().total == TaskBudget(max_cost_usd=1.0)


@pytest.mark.parametrize(("missing", "variable"), [
    ({"tts_price_per_mchar": None}, "TTS_PRICE_PER_MCHAR"),
    ({"image_price_per_image": None}, "IMAGE_PRICE_PER_IMAGE"),
    ({"search_price_per_request": None}, "SEARCH_PRICE_PER_REQUEST"),
    ({"llm_input_price_per_mtok": None, "llm_output_price_per_mtok": None}, "LLM model 'gpt-x'"),
    ({"llm_fallback_provider": "anthropic", "llm_fallback_model": "claude-x", "anthropic_api_key": "sk-ant-x123"},
     "LLM model 'claude-x'"),
])
def test_an_unpriced_paid_capability_stops_startup(missing: dict, variable: str) -> None:
    settings = production(**missing)
    with pytest.raises(ConfigError, match="run-wide spending cap cannot bind") as stopped:
        settings.validate_runtime()
    assert variable in str(stopped.value)


def test_without_the_cap_or_offline_nothing_more_is_required() -> None:
    uncapped = production(tts_price_per_mchar=None).model_copy(update={"budget_total_max_cost_usd": None})
    assert uncapped.cost_cap_problems() == []
    offline = Settings(providers=ProviderSettings(llm_routes={}), budget_total_max_cost_usd=1.0)
    assert offline.pricing_gaps() == [] and offline.cost_cap_problems() == []


@pytest.mark.parametrize("value", [math.inf, math.nan, -1.0])
def test_a_non_finite_or_negative_price_or_cap_is_invalid(value: float) -> None:
    with pytest.raises(ValidationError):
        ProviderSettings(llm_routes={}, tts_price_per_mchar=value)
    with pytest.raises(ValidationError):
        Settings(budget_total_max_cost_usd=value)


def test_a_price_for_no_model_and_half_an_llm_price_are_problems() -> None:
    lonely = ProviderSettings(llm_routes={}, llm_input_price_per_mtok=1.0, llm_output_price_per_mtok=2.0)
    assert any("prices LLM_MODEL, which is not set" in p for p in lonely.problems())
    half = ProviderSettings(llm_routes={}, llm_provider="mock", llm_model="m", llm_input_price_per_mtok=1.0)
    assert any("must be set together" in p for p in half.problems())


def test_environment_llm_pricing_charges_cached_input_at_the_input_price() -> None:
    from app.config.providers import apply_llm_overrides
    from app.config.routing import load_routing

    settings = production()
    routing = apply_llm_overrides(load_routing(settings.routing_file), settings.providers)
    assert routing.pricing["gpt-x"].cached_input_per_mtok == 2.5  # an upper bound of any vendor's cached price
