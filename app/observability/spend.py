"""Spend accounts: which budgets a provider attempt charges, and the store that keeps the ledger.

Every billable provider attempt is reserved against all the budget scopes of the account it runs under before it is
sent, and settled exactly once afterwards (see app/schemas/spend.py for the states). The SQL store
(`SqlSpendRepository`, storage layer) is the durable one; `InMemorySpendStore` gives code that runs without a
database (unit tests, throwaway scopes) the same semantics.

Scopes (`kind:id`): `task:{task_id}`, `session:{session_id}` (an interactive teaching session), `attempt:{attempt_id}`
(the grading of one assessment attempt through the API), `learner:{learner_id}:{YYYY-MM-DD}` and `global:{YYYY-MM-DD}`
(daily caps, UTC, only when configured), `local:{id}` (an in-memory ledger).

Reservations made for a task worker are fenced by its lease (a worker that lost it cannot start a paid call);
settlements never are: money already committed to a provider is recorded whatever happened to the lease.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from app.observability.budget import (
    METRICS,
    check_amount,
    exceeded,
    limit_for,
    micro,
    millis,
    reservation_amounts,
    settlement,
    _exceeded,
)
from app.schemas.common import ProviderRequestRecord, new_id, utcnow
from app.schemas.lease import Lease
from app.schemas.providers import Capability
from app.schemas.spend import (
    Charge,
    CounterTotals,
    Estimate,
    ReservationRefused,
    SettleResult,
    SpendEntry,
    SpendInFlight,
    SpendReservation,
    SpendSettlement,
    SpendState,
)
from app.schemas.usage import BudgetDefaults, TaskBudget, TaskUsage

logger = logging.getLogger(__name__)

SETTLE_TRIES = 3


class SpendStore(Protocol):
    def reserve(self, reservation: SpendReservation, *, fence: Lease | None, now: datetime) -> None:
        """Atomically add the reservation's charges to their counters and record the entry as reserved. Raises
        ReservationRefused (a counter has no room), LeaseLost (`fence` is not the task's current lease) or
        SpendInFlight (its idempotency key is reserved by another attempt); nothing is written then. Reserving an
        entry id that already exists changes nothing."""

    def settle(self, entry_id: str, settlement: SpendSettlement, *, now: datetime) -> SettleResult:
        """Move a reserved entry (or an `unknown` one, to `settled`) to its final state and adjust its counters, in
        one transaction. Never fenced. A repeated settlement changes nothing."""

    def expire_abandoned(self, now: datetime) -> list[str]:
        """Settle as `unknown` (charging their estimate) the reserved entries whose `reserved_until` has passed: their
        process died or could not record the outcome. Returns their ids."""

    def entries(self, *, task_id: str | None = None, session_id: str | None = None,
                attempt_ref: str | None = None) -> list[SpendEntry]: ...

    def totals(self, scope_keys: Iterable[str]) -> dict[str, dict[str, CounterTotals]]: ...

    def seed(self, scope_key: str, settled: dict[str, int], *, now: datetime) -> bool:
        """Create a scope's counters with spend recorded before the ledger existed, unless it already has any."""


def task_scope(task_id: str) -> str:
    return f"task:{task_id}"


def session_scope(session_id: str) -> str:
    return f"session:{session_id}"


def attempt_scope(attempt_id: str) -> str:
    return f"attempt:{attempt_id}"


def learner_day_scope(learner_id: str, now: datetime) -> str:
    return f"learner:{learner_id}:{now.date().isoformat()}"


def global_day_scope(now: datetime) -> str:
    return f"global:{now.date().isoformat()}"


def usage_counters(usage: TaskUsage) -> dict[str, int]:
    """Counter values equivalent to a TaskUsage (spend recorded before the ledger)."""
    return {"llm_requests": usage.llm_requests, "llm_tokens": usage.llm_tokens,
            "search_requests": usage.search_requests, "generated_images": usage.image_generations,
            "tts_characters": usage.tts_characters, "tts_ms": millis(usage.tts_seconds),
            "video_segments": usage.video_generations, "video_ms": millis(usage.video_generation_seconds),
            "video_cost_microusd": micro(usage.video_generation_cost_usd),
            "cost_microusd": micro(usage.estimated_cost_usd)}


@dataclass(frozen=True)
class BudgetScope:
    key: str
    budget: TaskBudget | None = None

    @property
    def kind(self) -> str:
        return scope_kind(self.key)


def scope_kind(key: str) -> str:
    """The budget a scope key names, as messages and events say it (an in-memory ledger stands for a task's)."""
    kind = key.split(":", 1)[0]
    return "task" if kind == "local" else kind


@dataclass(frozen=True)
class Reservation:
    entry_id: str
    capability: Capability
    operation: str
    amounts: dict[str, int]
    estimate: Estimate


@dataclass
class SpendAccount:
    """The budgets one piece of work spends against, the store that keeps its ledger, and how its entries are
    attributed. `fence`: the task worker's lease; reservations are refused once it is lost."""

    store: SpendStore
    scopes: tuple[BudgetScope, ...]
    task_id: str | None = None
    session_id: str | None = None
    attempt_ref: str | None = None
    learner_id: str | None = None
    fence: Lease | None = None
    clock: Callable[[], datetime] = utcnow
    grace_seconds: float = 60.0
    swept: list[str] = field(default_factory=list)  # abandoned reservations found by the last sweep (for events)

    @classmethod
    def local(cls, budget: TaskBudget | None = None, seed: TaskUsage | None = None) -> SpendAccount:
        """An in-memory account with one scope: what a ledger built without a store uses."""
        store = InMemorySpendStore()
        key = f"local:{new_id('acct')}"
        if seed is not None:
            store.seed(key, usage_counters(seed), now=utcnow())
        return cls(store=store, scopes=(BudgetScope(key, budget),))

    @property
    def scope_keys(self) -> list[str]:
        return [s.key for s in self.scopes]

    def reserve(self, *, entry_id: str, operation_id: str, attempt: int, capability: Capability, operation: str,
                provider: str, model: str | None, estimate: Estimate, timeout_seconds: float,
                node_id: str | None = None, agent_id: str | None = None,
                idempotency_key: str | None = None) -> Reservation | None:
        """Reserve one attempt before it is sent; None for an operation that is not billed. Raises
        BudgetExceededError (nothing is sent), LeaseLost or SpendInFlight."""
        amounts = reservation_amounts(capability, operation, estimate)
        if not amounts:
            return None
        charges = [Charge(scope_key=s.key, metric=m, amount=a, check=check_amount(m, a), limit=limit_for(s.budget, m))
                   for s in self.scopes for m, a in amounts.items()]
        now = self.clock()
        self.swept = self.store.expire_abandoned(now)
        reservation = SpendReservation(
            entry_id=entry_id, operation_id=operation_id, attempt=attempt, idempotency_key=idempotency_key,
            capability=capability.value, operation=operation, provider=provider, model=model, task_id=self.task_id,
            session_id=self.session_id, attempt_ref=self.attempt_ref, learner_id=self.learner_id, node_id=node_id,
            agent_id=agent_id, worker=self.fence.owner if self.fence else None,
            lease_fence=self.fence.fence if self.fence else None, charges=charges, estimate=amounts,
            estimated_cost_microusd=micro(estimate.cost_usd) if estimate.cost_usd is not None else None,
            reserved_until=now + timedelta(seconds=timeout_seconds + self.grace_seconds))
        try:
            self.store.reserve(reservation, fence=self.fence, now=now)
        except ReservationRefused as refused:
            c = refused.charge
            m = METRICS[c.metric]
            assert c.limit is not None
            raise _exceeded(m.limit, (refused.settled + refused.reserved + c.check) / m.scale, c.limit / m.scale,
                            f"{m.what.split(' (')[0]} of the {scope_kind(c.scope_key)} budget, with this "
                            "request", scope=scope_kind(c.scope_key)) from None
        return Reservation(entry_id=entry_id, capability=capability, operation=operation, amounts=amounts,
                           estimate=estimate)

    def settle(self, reservation: Reservation, record: ProviderRequestRecord | None,
               state: SpendState) -> SettleResult | None:
        """Record the attempt's outcome (bounded retries; when the store stays unavailable the entry is left
        reserved and the abandoned-reservation sweep charges its estimate). Never fenced."""
        s = settlement(reservation.capability, state, reservation.amounts, reservation.estimate, record)
        body = SpendSettlement(
            state=state, charged=s.charged, reported=s.reported, usage_source=s.usage_source,
            final_cost_microusd=s.final_cost_microusd, cost_source=s.cost_source,
            currency="USD" if s.final_cost_microusd is not None else None,
            vendor_request_id=record.vendor_request_id if record else None,
            error_type=record.error_type if record else None, latency_ms=record.latency_ms if record else None)
        for attempt in range(1, SETTLE_TRIES + 1):
            try:
                return self.store.settle(reservation.entry_id, body, now=self.clock())
            except Exception:  # noqa: BLE001 - the outcome must not be lost to a transient storage error
                if attempt == SETTLE_TRIES:
                    logger.exception("spend entry %s could not be settled; it stays reserved until the sweep "
                                     "charges its estimate", reservation.entry_id)
        return None

    def check_after(self, reservation: Reservation, result: SettleResult) -> None:
        """Raise BudgetExceededError when reported usage took an estimated metric over a limit (the attempt is
        recorded; the work stops here)."""
        for scope in self.scopes:
            for metric in reservation.amounts:
                limit = limit_for(scope.budget, metric)
                totals = result.totals.get(scope.key, {}).get(metric)
                if METRICS[metric].estimated and limit is not None and totals and totals.settled > limit:
                    raise exceeded(metric, totals.settled, limit, scope.kind)


# --- accounts for each kind of work --------------------------------------------------------------------------------


def _shared(defaults: BudgetDefaults, learner_id: str | None, now: datetime) -> list[BudgetScope]:
    scopes = []
    if defaults.learner_daily is not None and learner_id:
        scopes.append(BudgetScope(learner_day_scope(learner_id, now), defaults.learner_daily))
    if defaults.global_daily is not None:
        scopes.append(BudgetScope(global_day_scope(now), defaults.global_daily))
    return scopes


def task_account(store: SpendStore, defaults: BudgetDefaults, *, task_id: str, learner_id: str | None,
                 budget: TaskBudget | None, fence: Lease | None,
                 clock: Callable[[], datetime] = utcnow) -> SpendAccount:
    """A task worker's account: its task budget (the one in its metadata, else the default task budget), plus the
    learner's and the global daily caps when configured. Reservations are fenced by the worker's lease."""
    now = clock()
    scopes = (BudgetScope(task_scope(task_id), budget if budget is not None else defaults.task),
              *_shared(defaults, learner_id, now))
    return SpendAccount(store=store, scopes=scopes, task_id=task_id, learner_id=learner_id, fence=fence, clock=clock,
                        grace_seconds=defaults.reservation_grace_seconds)


def session_account(store: SpendStore, defaults: BudgetDefaults, *, session_id: str, task_id: str | None,
                    learner_id: str | None, clock: Callable[[], datetime] = utcnow) -> SpendAccount:
    """The account of a teaching session's teacher turns (the lesson task's own budget is not charged)."""
    now = clock()
    scopes = (BudgetScope(session_scope(session_id), defaults.session), *_shared(defaults, learner_id, now))
    return SpendAccount(store=store, scopes=scopes, task_id=task_id, session_id=session_id, learner_id=learner_id,
                        clock=clock, grace_seconds=defaults.reservation_grace_seconds)


def attempt_account(store: SpendStore, defaults: BudgetDefaults, *, attempt_id: str, task_id: str | None,
                    learner_id: str | None, session_id: str | None = None,
                    clock: Callable[[], datetime] = utcnow) -> SpendAccount:
    """The account of grading one assessment attempt outside a task: an answer submitted through the API, or a
    teaching session's answer (which also spends against the session's budget)."""
    now = clock()
    scopes = (BudgetScope(attempt_scope(attempt_id), defaults.attempt),
              *((BudgetScope(session_scope(session_id), defaults.session),) if session_id else ()),
              *_shared(defaults, learner_id, now))
    return SpendAccount(store=store, scopes=scopes, task_id=task_id, session_id=session_id, attempt_ref=attempt_id,
                        learner_id=learner_id, clock=clock, grace_seconds=defaults.reservation_grace_seconds)


# --- in-memory store ------------------------------------------------------------------------------------------------


_FROM = {SpendState.SETTLED: (SpendState.RESERVED, SpendState.UNKNOWN)}


@dataclass
class InMemorySpendStore:
    """The SpendStore semantics without a database (no lease storage: `fence` is not checked)."""

    _counters: dict[tuple[str, str], CounterTotals] = field(default_factory=dict)
    _entries: dict[str, SpendEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, reservation: SpendReservation, *, fence: Lease | None, now: datetime) -> None:
        with self._lock:
            if reservation.entry_id in self._entries:
                return
            key = reservation.idempotency_key
            if key and any(e.idempotency_key == key and e.state == SpendState.RESERVED for e in self._entries.values()):
                raise SpendInFlight(key)
            for c in reservation.charges:
                t = self._counters.get((c.scope_key, c.metric), CounterTotals())
                if c.limit is not None and t.settled + t.reserved + c.check > c.limit:
                    raise ReservationRefused(c, t.reserved, t.settled)
            for c in reservation.charges:
                t = self._counters.setdefault((c.scope_key, c.metric), CounterTotals())
                t.reserved += c.amount
            self._entries[reservation.entry_id] = SpendEntry(
                **reservation.model_dump(exclude={"charges"}),
                scope_keys=sorted({c.scope_key for c in reservation.charges}), state=SpendState.RESERVED,
                created_at=now)

    def settle(self, entry_id: str, settlement: SpendSettlement, *, now: datetime) -> SettleResult:
        with self._lock:
            entry = self._entries[entry_id]
            allowed = _FROM.get(settlement.state, (SpendState.RESERVED,))
            if entry.state not in allowed:
                same = entry.state == settlement.state and entry.charged == settlement.charged
                return SettleResult(outcome="duplicate" if same else "conflict",
                                    totals=self._totals(entry.scope_keys))
            self._apply(entry, settlement, now)
            return SettleResult(outcome="applied", totals=self._totals(entry.scope_keys))

    def _apply(self, entry: SpendEntry, s: SpendSettlement, now: datetime) -> None:
        previous = entry.state
        for key in entry.scope_keys:
            for metric in set(entry.estimate) | set(s.charged):
                t = self._counters.setdefault((key, metric), CounterTotals())
                if previous == SpendState.RESERVED:
                    t.reserved -= entry.estimate.get(metric, 0)
                    t.settled += s.charged.get(metric, 0)
                else:  # unknown -> settled: the charged estimate is replaced by the real usage
                    t.settled += s.charged.get(metric, 0) - entry.charged.get(metric, 0)
        self._entries[entry.entry_id] = entry.model_copy(update={
            **s.model_dump(), "settled_at": now})

    def expire_abandoned(self, now: datetime) -> list[str]:
        with self._lock:
            ids = [e.entry_id for e in self._entries.values()
                   if e.state == SpendState.RESERVED and e.reserved_until < now]
            for entry_id in ids:
                e = self._entries[entry_id]
                self._apply(e, abandoned(e), now)
            return ids

    def entries(self, *, task_id: str | None = None, session_id: str | None = None,
                attempt_ref: str | None = None) -> list[SpendEntry]:
        return [e for e in self._entries.values()
                if (task_id is None or e.task_id == task_id) and (session_id is None or e.session_id == session_id)
                and (attempt_ref is None or e.attempt_ref == attempt_ref)]

    def totals(self, scope_keys: Iterable[str]) -> dict[str, dict[str, CounterTotals]]:
        with self._lock:
            return self._totals(scope_keys)

    def _totals(self, scope_keys: Iterable[str]) -> dict[str, dict[str, CounterTotals]]:
        keys = set(scope_keys)
        out: dict[str, dict[str, CounterTotals]] = {}
        for (key, metric), t in self._counters.items():
            if key in keys:
                out.setdefault(key, {})[metric] = t.model_copy()
        return out

    def seed(self, scope_key: str, settled: dict[str, int], *, now: datetime) -> bool:
        with self._lock:
            if any(k == scope_key for k, _ in self._counters):
                return False
            for metric, amount in settled.items():
                self._counters[(scope_key, metric)] = CounterTotals(settled=amount)
            return True


def abandoned(entry: SpendEntry) -> SpendSettlement:
    """The settlement of a reservation nobody settled in time: it may have been billed, so its estimate is charged
    (marked as an estimate, never as provider-reported usage)."""
    cost = entry.estimated_cost_microusd
    return SpendSettlement(state=SpendState.UNKNOWN, charged=dict(entry.estimate), usage_source="estimate",
                           final_cost_microusd=cost, cost_source="estimate" if cost is not None else "unpriced",
                           currency="USD" if cost is not None else None, error_type="Abandoned")
