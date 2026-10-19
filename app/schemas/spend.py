"""The spend ledger: one durable entry per billable provider attempt, and the budget counters it charges.

An attempt is *reserved* before anything is sent (its estimate is added to every counter of the account it runs
under, and refused if a limit would be crossed), then *settled* once: `settled` (the provider answered), `failed`
(the provider refused it; only request counts are charged), `unknown` (it may have been billed: a timeout, a
cancellation, an unusable answer, or a reservation abandoned by a crashed process; its estimate is charged) or
`released` (it was never sent). The entry id is our own request id, generated before the reservation, never a
provider's id. Entries hold ids and numbers only: no prompt, answer, learner text or provider payload.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from app.schemas.common import Schema


class SpendState(str, Enum):
    RESERVED = "reserved"
    SETTLED = "settled"
    FAILED = "failed"
    UNKNOWN = "unknown"
    RELEASED = "released"


# Where an entry's charged usage comes from: what the provider reported, our pre-call estimate, or reported fields
# completed with the estimate for the ones it did not report.
UsageSource = Literal["reported", "partial", "estimate", "none"]
# Where its cost comes from: the provider's own figure, configured pricing applied to reported usage, configured
# pricing applied to the estimate (never presented as provider-reported), or nothing known.
CostSource = Literal["reported", "priced", "estimate", "unpriced", "none"]


class Estimate(Schema):
    """What one attempt is expected to use, known before it is sent. None: not known (charged as zero)."""

    llm_tokens: int | None = Field(default=None, ge=0)  # input estimate + the request's max output tokens
    characters: int | None = Field(default=None, ge=0)
    audio_seconds: float | None = Field(default=None, ge=0)
    video_seconds: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)  # configured pricing applied to the estimate


class Charge(Schema):
    """One counter a reservation must fit into: `settled + reserved + check <= limit` (no limit: always fits).
    `amount` is added to the counter's reservation; `check` is at least `amount` (one unit of headroom is required
    for estimated metrics, so a limit that is used up stops the next request even when its estimate is zero)."""

    scope_key: str
    metric: str
    amount: int = Field(ge=0)
    check: int = Field(ge=0)
    limit: int | None = None


class SpendReservation(Schema):
    entry_id: str  # our request id for this attempt (preq_...)
    operation_id: str  # one per logical provider operation, shared by its retries
    attempt: int = Field(ge=1)
    idempotency_key: str | None = None  # at most one reserved (in-flight) attempt per key
    capability: str
    operation: str
    provider: str
    model: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    attempt_ref: str | None = None  # an assessment attempt id
    learner_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    worker: str | None = None  # the task lease owner that reserved it
    lease_fence: int | None = None
    charges: list[Charge] = Field(default_factory=list)
    estimate: dict[str, int] = Field(default_factory=dict)  # metric -> amount reserved
    estimated_cost_microusd: int | None = None
    reserved_until: datetime


class SpendSettlement(Schema):
    state: SpendState
    charged: dict[str, int] = Field(default_factory=dict)  # metric -> amount the counters keep for this attempt
    reported: dict[str, float] | None = None  # provider-reported units (tokens, characters, seconds, images)
    usage_source: UsageSource = "none"
    final_cost_microusd: int | None = None
    cost_source: CostSource = "none"
    currency: str | None = None
    vendor_request_id: str | None = None
    error_type: str | None = None  # a class name, never a message
    latency_ms: float | None = None


class SpendEntry(Schema):
    entry_id: str
    operation_id: str
    attempt: int
    idempotency_key: str | None = None
    capability: str
    operation: str
    provider: str
    model: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    attempt_ref: str | None = None
    learner_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    worker: str | None = None
    lease_fence: int | None = None
    scope_keys: list[str] = Field(default_factory=list)
    state: SpendState
    estimate: dict[str, int] = Field(default_factory=dict)
    charged: dict[str, int] = Field(default_factory=dict)
    reported: dict[str, float] | None = None
    usage_source: UsageSource = "none"
    estimated_cost_microusd: int | None = None
    final_cost_microusd: int | None = None
    cost_source: CostSource = "none"
    currency: str | None = None
    vendor_request_id: str | None = None
    error_type: str | None = None
    latency_ms: float | None = None
    created_at: datetime
    settled_at: datetime | None = None
    reserved_until: datetime


class CounterTotals(Schema):
    reserved: int = 0
    settled: int = 0


class SettleResult(Schema):
    """`applied`: this call changed the entry; `duplicate`: the same settlement was already stored; `conflict`: a
    different one was (nothing changed). `totals` are the account's counters after the call, by scope and metric.
    `stale_lease`: the entry was reserved under a task lease that is no longer current (spend kept regardless)."""

    outcome: Literal["applied", "duplicate", "conflict"]
    totals: dict[str, dict[str, CounterTotals]] = Field(default_factory=dict)
    stale_lease: bool = False


class ReservationRefused(Exception):
    """A counter of the account had no room for the reservation. Nothing was reserved or recorded."""

    def __init__(self, charge: Charge, reserved: int, settled: int) -> None:
        super().__init__(f"{charge.scope_key}/{charge.metric}: {settled} settled + {reserved} reserved + "
                         f"{charge.check} > {charge.limit}")
        self.charge = charge
        self.reserved = reserved
        self.settled = settled


class SpendInFlight(Exception):
    """Another attempt with the same idempotency key is reserved and not yet settled: this one was not sent."""

    def __init__(self, idempotency_key: str) -> None:
        super().__init__(f"an attempt for this operation is already in flight ({idempotency_key})")
        self.idempotency_key = idempotency_key
