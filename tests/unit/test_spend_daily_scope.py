"""Daily budget scopes across midnight UTC (F8): the learner's and the global day are resolved for each reservation,
not when the account is built, and an attempt that crosses midnight is charged once, entirely to the day it was
reserved on, through settlement, the abandoned-reservation sweep and a late real result. For the SQL store (SQLite)
and the in-memory one, with a controlled clock: no timing, no sleeps."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.config.settings import Settings
from app.observability.budget import BudgetExceededError
from app.observability.scope import UsageLedger
from app.observability.spend import (
    InMemorySpendStore,
    attempt_account,
    global_day_scope,
    learner_day_scope,
    session_account,
    task_account,
)
from app.schemas.common import ProviderRequestRecord, new_id
from app.schemas.providers import Capability
from app.schemas.spend import Estimate, SpendState
from app.schemas.usage import BudgetDefaults, TaskBudget
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlSpendRepository

BEFORE = datetime(2026, 10, 18, 23, 59, 30, tzinfo=timezone.utc)
DAY1, DAY2 = "2026-10-18", "2026-10-19"


class Clock:
    def __init__(self, now: datetime = BEFORE) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture(params=["sql", "memory"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemorySpendStore()
        return
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    yield SqlSpendRepository(sessions)
    dispose(sessions)


def reserve(account, tokens: int = 10, timeout: float = 60):
    n = new_id("preq")
    return account.reserve(entry_id=n, operation_id=f"pop_{n}", attempt=1, capability=Capability.LLM,
                           operation="generate", provider="p", model="m", estimate=Estimate(llm_tokens=tokens),
                           timeout_seconds=timeout)


def record(input_tokens: int, output_tokens: int) -> ProviderRequestRecord:
    return ProviderRequestRecord(request_id=new_id("preq"), provider="p", capability="llm", operation="generate",
                                 input_tokens=input_tokens, output_tokens=output_tokens)


def counter(store, key: str, metric: str = "llm_tokens") -> tuple[int, int]:
    t = store.totals([key]).get(key, {}).get(metric)
    return (t.reserved, t.settled) if t else (0, 0)


def caps(learner: TaskBudget | None = None, glob: TaskBudget | None = None) -> BudgetDefaults:
    return BudgetDefaults(learner_daily=learner, global_daily=glob)


L1, L2 = learner_day_scope("L1", BEFORE), learner_day_scope("L1", BEFORE + timedelta(minutes=1))
G1, G2 = global_day_scope(BEFORE), global_day_scope(BEFORE + timedelta(minutes=1))


def test_day_keys_are_utc_calendar_days() -> None:
    assert (L1, L2, G1, G2) == (f"learner:L1:{DAY1}", f"learner:L1:{DAY2}", f"global:{DAY1}", f"global:{DAY2}")
    tokyo = timezone(timedelta(hours=9))
    # 08:30 on the 19th in Tokyo is 23:30 on the 18th in UTC: the UTC day counts, never the clock's own zone.
    assert global_day_scope(datetime(2026, 10, 19, 8, 30, tzinfo=tokyo)) == f"global:{DAY1}"
    assert global_day_scope(datetime(2026, 10, 18, 23, 59, 59, 999999, tzinfo=timezone.utc)) == f"global:{DAY1}"
    assert global_day_scope(datetime(2026, 10, 19, 0, 0, tzinfo=timezone.utc)) == f"global:{DAY2}"
    assert global_day_scope(datetime(2026, 10, 18, 23, 59)) == f"global:{DAY1}"  # naive: taken as UTC


def test_an_account_built_before_midnight_charges_the_new_day_after_it(store) -> None:
    """The F8 reproduction: a task worker started at 23:59:30 kept charging (and being refused by) the previous day.
    Now the new day has its own room; the task budget spans both days."""
    clock = Clock()
    acct = task_account(store, caps(TaskBudget(max_llm_requests=1), TaskBudget(max_llm_requests=5)), task_id="t1",
                        learner_id="L1", budget=TaskBudget(max_llm_requests=3), fence=None, clock=clock)
    assert acct.scope_keys == ["task:t1", L1, G1]
    reserve(acct)
    with pytest.raises(BudgetExceededError) as e:
        reserve(acct)
    assert e.value.scope == "learner"
    clock.advance(seconds=31)  # 00:00:01 the next UTC day, same account
    assert acct.scope_keys == ["task:t1", L2, G2]
    reserve(acct)
    assert [counter(store, k, "llm_requests") for k in (L1, L2, G1, G2)] == [(1, 0), (1, 0), (1, 0), (1, 0)]
    assert counter(store, "task:t1", "llm_requests") == (2, 0)
    with pytest.raises(BudgetExceededError) as e:
        reserve(acct)
    assert e.value.scope == "learner"
    [day1, day2] = sorted(store.entries(task_id="t1"), key=lambda e: e.created_at)
    assert (day1.scope_keys, day2.scope_keys) == (sorted(["task:t1", L1, G1]), sorted(["task:t1", L2, G2]))


@pytest.mark.parametrize("build", ["task", "session", "attempt"])
def test_every_kind_of_account_resolves_its_day_per_reservation(store, build) -> None:
    clock = Clock()
    d = caps(TaskBudget(max_llm_requests=1))
    acct = {"task": lambda: task_account(store, d, task_id="t1", learner_id="L1", budget=None, fence=None,
                                         clock=clock),
            "session": lambda: session_account(store, d, session_id="s1", task_id="t1", learner_id="L1", clock=clock),
            "attempt": lambda: attempt_account(store, d, attempt_id="a1", task_id="t1", learner_id="L1",
                                               session_id="s1", clock=clock)}[build]()
    reserve(acct)
    clock.advance(minutes=1)
    reserve(acct)  # refused before F8: the account kept the 18th
    assert counter(store, L1, "llm_requests") == counter(store, L2, "llm_requests") == (1, 0)


def test_a_settlement_after_midnight_stays_on_the_day_it_was_reserved(store) -> None:
    """Reserved at 23:59:30, answered at 00:00:30: the reported usage replaces the estimate on the 18th's counters
    (where it was reserved), and the 19th is untouched. Reserved and settled always pair on the same counter."""
    clock = Clock()
    acct = task_account(store, caps(TaskBudget(max_llm_tokens=1000), TaskBudget()), task_id="t1", learner_id="L1",
                        budget=None, fence=None, clock=clock)
    r = reserve(acct, tokens=400)
    clock.advance(minutes=1)
    result = acct.settle(r, record(100, 50), SpendState.SETTLED)
    assert result is not None and result.outcome == "applied"
    assert counter(store, L1) == counter(store, G1) == counter(store, "task:t1") == (0, 150)
    assert counter(store, L2) == counter(store, G2) == (0, 0)
    assert set(result.totals) == {"task:t1", L1, G1}
    [entry] = store.entries(task_id="t1")
    naive = lambda t: t.replace(tzinfo=None)  # noqa: E731 - SQLite returns these timestamps without a zone
    assert (naive(entry.created_at), naive(entry.settled_at)) == (naive(BEFORE), naive(clock.now))


def test_usage_over_a_daily_cap_is_checked_on_the_reservations_day(store) -> None:
    """A response that reports more than its estimate stops the work when it takes the day it was reserved on over
    the cap, even though the settlement happens on the next day (whose counters are empty)."""
    clock = Clock()
    acct = task_account(store, caps(TaskBudget(max_llm_tokens=100)), task_id="t1", learner_id="L1", budget=None,
                        fence=None, clock=clock)
    ledger = UsageLedger(account=acct)
    r = ledger.reserve(entry_id="preq_1", operation_id="pop_1", attempt=1, capability=Capability.LLM,
                       operation="generate", provider="p", model="m", estimate=Estimate(llm_tokens=50),
                       timeout_seconds=60)
    clock.advance(minutes=1)
    with pytest.raises(BudgetExceededError) as e:
        ledger.settle(r, record(100, 50), SpendState.SETTLED)
    assert e.value.scope == "learner"
    assert counter(store, L1) == (0, 150) and counter(store, L2) == (0, 0)


def test_an_abandoned_reservation_is_swept_and_corrected_on_the_day_it_was_reserved(store) -> None:
    """A worker that reserved at 23:59:30 and died: the sweep run by the next reservation (on the 19th) charges the
    estimate to the 18th; a late real result then replaces it there, once. The 19th holds only its own reservation."""
    clock = Clock()
    d = caps(TaskBudget(max_llm_tokens=1000), TaskBudget(max_llm_tokens=10_000))
    dead = task_account(store, d, task_id="t1", learner_id="L1", budget=None, fence=None, clock=clock)
    r = reserve(dead, tokens=400, timeout=30)  # reserved_until: 30 s + 60 s grace, i.e. 00:01:00 on the 19th
    clock.advance(minutes=2)
    other = session_account(store, d, session_id="s1", task_id=None, learner_id="L1", clock=clock)
    reserve(other, tokens=200)
    assert other.swept == [r.entry_id]
    assert counter(store, L1) == counter(store, G1) == (0, 400)
    assert counter(store, L2) == counter(store, G2) == (200, 0)
    entry = next(e for e in store.entries(task_id="t1"))
    assert (entry.state, entry.usage_source) == (SpendState.UNKNOWN, "estimate")

    late = dead.settle(r, record(100, 50), SpendState.SETTLED)
    assert late is not None and late.outcome == "applied"
    assert counter(store, L1) == counter(store, G1) == counter(store, "task:t1") == (0, 150)
    assert counter(store, L2) == counter(store, G2) == (200, 0)
    assert dead.settle(r, record(100, 50), SpendState.SETTLED).outcome == "duplicate"
    assert counter(store, L1) == (0, 150)


def test_released_and_failed_attempts_across_midnight_return_their_room_to_their_own_day(store) -> None:
    clock = Clock()
    acct = task_account(store, caps(TaskBudget(max_llm_requests=1, max_llm_tokens=100)), task_id="t1",
                        learner_id="L1", budget=None, fence=None, clock=clock)
    released = reserve(acct, tokens=50)
    clock.advance(seconds=10)  # 23:59:40, still the 18th
    acct.settle(released, None, SpendState.RELEASED)
    failed = reserve(acct, tokens=50)
    clock.advance(minutes=1)
    acct.settle(failed, None, SpendState.FAILED)
    assert counter(store, L1) == (0, 0) and counter(store, L1, "llm_requests") == (0, 1)
    assert counter(store, L2, "llm_requests") == (0, 0)
    reserve(acct)  # the 19th's single request is free


def test_daily_caps_are_off_by_default_and_other_budgets_are_unchanged(tmp_path) -> None:
    defaults = Settings(data_dir=tmp_path).budget_defaults()
    assert defaults.learner_daily is None and defaults.global_daily is None
    store, clock = InMemorySpendStore(), Clock()
    accounts = [task_account(store, defaults, task_id="t1", learner_id="L1", budget=None, fence=None, clock=clock),
                session_account(store, defaults, session_id="s1", task_id="t1", learner_id="L1", clock=clock),
                attempt_account(store, defaults, attempt_id="a1", task_id="t1", learner_id="L1", session_id="s1",
                                clock=clock)]
    assert [a.scope_keys for a in accounts] == [["task:t1"], ["session:s1"], ["attempt:a1", "session:s1"]]
    clock.advance(minutes=1)
    assert [a.scope_keys for a in accounts] == [["task:t1"], ["session:s1"], ["attempt:a1", "session:s1"]]
    for a in accounts:
        r = reserve(a)
        assert {s.key for s in r.scopes} == set(a.scope_keys)
    assert not any(k.startswith(("learner:", "global:")) for e in store.entries() for k in e.scope_keys)
    # A daily cap without a learner id applies the global day only.
    acct = task_account(store, caps(TaskBudget(), TaskBudget()), task_id="t2", learner_id=None, budget=None,
                        fence=None, clock=clock)
    assert acct.scope_keys == ["task:t2", G2]


# --- concurrency ----------------------------------------------------------------------------------------------------

PER_DAY = 7
WORKERS = 6
EACH = 6  # 36 attempts, half on each day: more than either day's cap


def _race(url: str, learner: str, start: threading.Barrier, out: list) -> None:
    """One worker with its own engine and account. Its clock alternates between 23:59:59.9 and 00:00:00.1, so every
    worker crosses midnight back and forth while the others do too."""
    sessions = create_db(url)
    store = SqlSpendRepository(sessions)
    times = [BEFORE + timedelta(seconds=29.9), BEFORE + timedelta(seconds=30.1)]
    clock = Clock()
    acct = task_account(store, caps(TaskBudget(max_llm_requests=PER_DAY)), task_id=new_id("task"),
                        learner_id=learner, budget=None, fence=None, clock=clock)
    admitted = {DAY1: 0, DAY2: 0}
    store.totals(["x"])
    start.wait()
    try:
        for i in range(EACH):
            clock.now = times[i % 2]
            try:
                r = reserve(acct)
            except BudgetExceededError:
                continue
            admitted[(DAY1, DAY2)[i % 2]] += 1
            if i % 3 == 0:  # some settled at once, on the other side of midnight
                clock.now = times[(i + 1) % 2]
                acct.settle(r, record(5, 5), SpendState.SETTLED)
    finally:
        dispose(sessions)
    out.append(admitted)


def check_days(url: str, learner: str, results: list[dict]) -> None:
    sessions = create_db(url)
    store = SqlSpendRepository(sessions)
    try:
        for day in (DAY1, DAY2):
            key = f"learner:{learner}:{day}"
            assert sum(r[day] for r in results) == PER_DAY  # each day admits exactly its cap, no more, no fewer
            entries = [e for e in store.entries() if key in e.scope_keys]
            assert len(entries) == PER_DAY
            t = store.totals([key])[key]["llm_requests"]
            assert t.reserved + t.settled == PER_DAY
            assert t.settled == sum(e.charged.get("llm_requests", 0) for e in entries
                                    if e.state != SpendState.RESERVED)
            assert t.reserved == sum(e.estimate.get("llm_requests", 0) for e in entries
                                     if e.state == SpendState.RESERVED)
    finally:
        dispose(sessions)


def race_across_midnight(url: str) -> None:
    learner = new_id("learner")
    dispose(create_db(url))
    results: list[dict] = []
    start = threading.Barrier(WORKERS)
    threads = [threading.Thread(target=_race, args=(url, learner, start, results)) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert len(results) == WORKERS
    check_days(url, learner, results)


def test_concurrent_workers_crossing_midnight_never_exceed_either_days_cap(tmp_path) -> None:
    race_across_midnight(f"sqlite:///{tmp_path / 'db.sqlite'}")
