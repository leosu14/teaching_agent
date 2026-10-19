"""Budget enforcement for one task's provider spending.

The provider layer asks before every billable request (`admit`) and reports every request it made (`check_after`).
A request that would exceed a request or unit limit is never sent; a limit that is only known once a provider has
answered (tokens, audio seconds, cost) stops the task at the request that reaches it. Either way the task fails with
BudgetExceededError, which is never retried and never falls back to another provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.common import ProviderRequestRecord
from app.schemas.providers import Capability
from app.schemas.spend import Estimate, SpendState
from app.schemas.usage import BILLABLE, TaskBudget, TaskUsage


class BudgetExceededError(Exception):
    """A task reached one of its budget limits. Not a provider failure: retrying or falling back would spend more."""

    def __init__(self, message: str, *, limit: str, used: float, maximum: float, scope: str = "task") -> None:
        super().__init__(message)
        self.limit = limit
        self.used = used
        self.maximum = maximum
        self.scope = scope  # the kind of budget: task, session, attempt, learner or global (never an id)


def _exceeded(limit: str, used: float, maximum: float, what: str, *, scope: str = "task") -> BudgetExceededError:
    after = ("the task stops here. Raise the limit and resume the task to continue." if scope == "task" else
             f"no further provider request is made against this {scope} budget until its limit is raised"
             + (" or the next day starts." if scope in ("learner", "global") else "."))
    return BudgetExceededError(f"budget exceeded: {what} (limit {limit}={maximum:g}, used {used:g}); {after}",
                               limit=limit, used=used, maximum=maximum, scope=scope)


def admit(budget: TaskBudget, usage: TaskUsage, capability: Capability, operation: str,
          units: dict[str, float] | None = None) -> None:
    """Raise BudgetExceededError if sending this request would exceed the budget."""
    if operation not in BILLABLE.get(capability, ()):
        return
    units = units or {}
    if capability == Capability.LLM:
        if budget.max_llm_requests is not None and usage.llm_requests + 1 > budget.max_llm_requests:
            raise _exceeded("MAX_LLM_REQUESTS", usage.llm_requests + 1, budget.max_llm_requests,
                            "one more LLM request")
        if budget.max_llm_tokens is not None and usage.llm_tokens >= budget.max_llm_tokens:
            raise _exceeded("MAX_LLM_TOKENS", usage.llm_tokens, budget.max_llm_tokens, "LLM tokens already used up")
    elif capability == Capability.SEARCH:
        if budget.max_search_requests is not None and usage.search_requests + 1 > budget.max_search_requests:
            raise _exceeded("MAX_SEARCH_REQUESTS", usage.search_requests + 1, budget.max_search_requests,
                            "one more search request")
    elif capability == Capability.IMAGE:
        if budget.max_generated_images is not None and usage.image_generations + 1 > budget.max_generated_images:
            raise _exceeded("MAX_GENERATED_IMAGES", usage.image_generations + 1, budget.max_generated_images,
                            "one more image generation")
    elif capability == Capability.TTS:
        characters = int(units.get("characters", 0))
        if budget.max_tts_characters is not None and usage.tts_characters + characters > budget.max_tts_characters:
            raise _exceeded("MAX_TTS_CHARACTERS", usage.tts_characters + characters, budget.max_tts_characters,
                            f"speech synthesis of {characters} more characters")
        if budget.max_tts_seconds is not None and usage.tts_seconds >= budget.max_tts_seconds:
            raise _exceeded("MAX_TTS_SECONDS", usage.tts_seconds, budget.max_tts_seconds, "TTS audio already used up")
    elif capability == Capability.VIDEO_GENERATION:
        seconds = float(units.get("seconds", 0.0))
        cost = units.get("cost_usd")
        if (budget.max_generated_video_segments is not None
                and usage.video_generations + 1 > budget.max_generated_video_segments):
            raise _exceeded("MAX_GENERATED_VIDEO_SEGMENTS", usage.video_generations + 1,
                            budget.max_generated_video_segments, "one more generated video segment")
        if (budget.max_generated_video_seconds is not None
                and usage.video_generation_seconds + seconds > budget.max_generated_video_seconds + 1e-9):
            raise _exceeded("MAX_GENERATED_VIDEO_SECONDS", round(usage.video_generation_seconds + seconds, 3),
                            budget.max_generated_video_seconds, f"{seconds:g} more generated video seconds")
        if (budget.max_video_generation_cost_usd is not None and cost is not None
                and usage.video_generation_cost_usd + cost > budget.max_video_generation_cost_usd + 1e-9):
            raise _exceeded("MAX_VIDEO_GENERATION_COST_USD", round(usage.video_generation_cost_usd + cost, 6),
                            budget.max_video_generation_cost_usd, f"video generation costing ${cost:g}")
    if budget.max_cost_usd is not None and usage.estimated_cost_usd >= budget.max_cost_usd:
        raise _exceeded("MAX_COST_USD", usage.estimated_cost_usd, budget.max_cost_usd, "known provider cost used up")


def check_after(budget: TaskBudget, usage: TaskUsage) -> None:
    """Raise BudgetExceededError if what providers reported so far is over a limit that could not be checked in
    advance (tokens, audio seconds, cost)."""
    if budget.max_llm_tokens is not None and usage.llm_tokens > budget.max_llm_tokens:
        raise _exceeded("MAX_LLM_TOKENS", usage.llm_tokens, budget.max_llm_tokens, "LLM tokens")
    if budget.max_tts_seconds is not None and usage.tts_seconds > budget.max_tts_seconds:
        raise _exceeded("MAX_TTS_SECONDS", usage.tts_seconds, budget.max_tts_seconds, "TTS audio seconds")
    if budget.max_cost_usd is not None and usage.estimated_cost_usd > budget.max_cost_usd:
        raise _exceeded("MAX_COST_USD", usage.estimated_cost_usd, budget.max_cost_usd, "known provider cost")


# --- the spend ledger's counters ------------------------------------------------------------------------------------
#
# A budget limit is enforced on integer counters, one per (scope, metric), in fixed units so no float tolerance is
# needed. A metric is "exact" when its amount is known before the request (requests, characters, video seconds) and
# "estimated" when only the provider's answer gives it (tokens, audio seconds, cost): such a metric reserves an
# estimate and requires one unit of headroom, so a limit that is used up stops the next request even when the
# estimate is zero (the same rule as `admit` above).


@dataclass(frozen=True)
class Metric:
    field: str  # the TaskBudget limit
    limit: str  # its name in messages (the production variable)
    scale: int  # counter units per limit unit
    estimated: bool
    what: str


METRICS: dict[str, Metric] = {
    "llm_requests": Metric("max_llm_requests", "MAX_LLM_REQUESTS", 1, False, "LLM requests"),
    "llm_tokens": Metric("max_llm_tokens", "MAX_LLM_TOKENS", 1, True, "LLM tokens"),
    "search_requests": Metric("max_search_requests", "MAX_SEARCH_REQUESTS", 1, False, "search requests"),
    "generated_images": Metric("max_generated_images", "MAX_GENERATED_IMAGES", 1, False, "image generations"),
    "tts_characters": Metric("max_tts_characters", "MAX_TTS_CHARACTERS", 1, False, "TTS characters"),
    "tts_ms": Metric("max_tts_seconds", "MAX_TTS_SECONDS", 1000, True, "TTS audio (ms)"),
    "video_segments": Metric("max_generated_video_segments", "MAX_GENERATED_VIDEO_SEGMENTS", 1, False,
                             "generated video segments"),
    "video_ms": Metric("max_generated_video_seconds", "MAX_GENERATED_VIDEO_SECONDS", 1000, False,
                       "generated video (ms)"),
    "video_cost_microusd": Metric("max_video_generation_cost_usd", "MAX_VIDEO_GENERATION_COST_USD", 1_000_000, False,
                                  "video generation cost (micro-USD)"),
    "cost_microusd": Metric("max_cost_usd", "MAX_COST_USD", 1_000_000, True, "known provider cost (micro-USD)"),
}
# Metrics a failed attempt still charges: a refused request counts as a request (failed attempts always counted).
COUNT_METRICS = frozenset({"llm_requests", "search_requests", "generated_images", "video_segments"})


def micro(usd: float) -> int:
    return int(round(usd * 1_000_000))


def millis(seconds: float) -> int:
    return int(round(seconds * 1000))


def limit_for(budget: TaskBudget | None, metric: str) -> int | None:
    """The budget's limit for a counter metric, in counter units; None = no limit."""
    if budget is None:
        return None
    m = METRICS[metric]
    value = getattr(budget, m.field)
    return None if value is None else int(math.floor(value * m.scale + 1e-9))


def reservation_amounts(capability: Capability, operation: str, estimate: Estimate) -> dict[str, int]:
    """What a billable attempt reserves, per metric ({} for an operation that is not billed)."""
    if operation not in BILLABLE.get(capability, ()):
        return {}
    cost = micro(estimate.cost_usd) if estimate.cost_usd is not None else 0
    if capability == Capability.LLM:
        return {"llm_requests": 1, "llm_tokens": estimate.llm_tokens or 0, "cost_microusd": cost}
    if capability == Capability.SEARCH:
        return {"search_requests": 1, "cost_microusd": cost}
    if capability == Capability.IMAGE:
        return {"generated_images": 1, "cost_microusd": cost}
    if capability == Capability.TTS:
        return {"tts_characters": estimate.characters or 0, "tts_ms": millis(estimate.audio_seconds or 0.0),
                "cost_microusd": cost}
    if capability == Capability.VIDEO_GENERATION:
        return {"video_segments": 1, "video_ms": millis(estimate.video_seconds or 0.0), "video_cost_microusd": cost,
                "cost_microusd": cost}
    return {"cost_microusd": cost}  # image search: no unit limit exists, only cost


def check_amount(metric: str, amount: int) -> int:
    return max(amount, 1) if METRICS[metric].estimated else amount


@dataclass(frozen=True)
class Settled:
    """What a settled attempt charges and how much of it the provider reported."""

    charged: dict[str, int]
    reported: dict[str, float] | None
    usage_source: str
    final_cost_microusd: int | None
    cost_source: str


def settlement(capability: Capability, state: SpendState, reserved: dict[str, int], estimate: Estimate,
               record: ProviderRequestRecord | None) -> Settled:
    """The charge of an attempt in its final state. `reserved` is what it reserved (its estimate)."""
    estimated_cost = micro(estimate.cost_usd) if estimate.cost_usd is not None else None
    if state == SpendState.RELEASED:
        return Settled({}, None, "none", None, "none")
    if state == SpendState.FAILED:
        return Settled({m: a for m, a in reserved.items() if m in COUNT_METRICS}, None, "none", None, "none")
    if state == SpendState.UNKNOWN or record is None:  # it may have been billed: charge the estimate
        return Settled(dict(reserved), None, "estimate", estimated_cost,
                       "estimate" if estimated_cost is not None else "unpriced")
    charged = dict(reserved)
    reported = {k: float(v) for k, v in (
        ("input_tokens", record.input_tokens), ("output_tokens", record.output_tokens),
        ("characters", record.characters), ("audio_seconds", record.audio_seconds),
        ("video_seconds", record.video_seconds), ("image_count", record.image_count),
        ("result_count", record.result_count)) if v is not None}
    estimated: list[str] = []
    if "llm_tokens" in charged:
        tokens = (record.input_tokens or 0) + (record.output_tokens or 0)
        if tokens > 0:
            charged["llm_tokens"] = tokens
        else:  # no usage reported (a real model call never uses zero tokens): the estimate stands
            estimated.append("llm_tokens")
    if "tts_characters" in charged:
        if record.characters is not None:
            charged["tts_characters"] = record.characters
        else:
            estimated.append("tts_characters")
    if "tts_ms" in charged:
        if record.audio_seconds is not None:
            charged["tts_ms"] = millis(record.audio_seconds)
        else:
            estimated.append("tts_ms")
    if "video_ms" in charged and record.video_seconds is not None:
        charged["video_ms"] = millis(record.video_seconds)
    if record.actual_cost_usd is not None:
        cost, source = micro(record.actual_cost_usd), "reported"
    elif record.estimated_cost_usd is not None and "llm_tokens" not in estimated:
        cost, source = micro(record.estimated_cost_usd), "priced"
    elif estimated_cost:
        cost, source = estimated_cost, "estimate"
    else:
        cost, source = None, "unpriced"
    for metric in ("cost_microusd", "video_cost_microusd"):
        if metric in charged:
            charged[metric] = cost or 0
    usage_source = "partial" if estimated else "reported"
    if estimated and not reported:
        usage_source = "estimate"
    return Settled(charged, reported or None, usage_source, cost, source)


def exceeded(metric: str, settled: int, limit: int, scope: str) -> BudgetExceededError:
    m = METRICS[metric]
    return _exceeded(m.limit, settled / m.scale, limit / m.scale, f"{m.what.split(' (')[0]} of the {scope} budget",
                     scope=scope)
