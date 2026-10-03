"""Budget enforcement for one task's provider spending.

The provider layer asks before every billable request (`admit`) and reports every request it made (`check_after`).
A request that would exceed a request or unit limit is never sent; a limit that is only known once a provider has
answered (tokens, audio seconds, cost) stops the task at the request that reaches it. Either way the task fails with
BudgetExceededError, which is never retried and never falls back to another provider.
"""

from __future__ import annotations

from app.schemas.providers import Capability
from app.schemas.usage import BILLABLE, TaskBudget, TaskUsage


class BudgetExceededError(Exception):
    """A task reached one of its budget limits. Not a provider failure: retrying or falling back would spend more."""

    def __init__(self, message: str, *, limit: str, used: float, maximum: float) -> None:
        super().__init__(message)
        self.limit = limit
        self.used = used
        self.maximum = maximum


def _exceeded(limit: str, used: float, maximum: float, what: str) -> BudgetExceededError:
    return BudgetExceededError(f"budget exceeded: {what} (limit {limit}={maximum:g}, used {used:g}); the task stops "
                               "here. Raise the limit and resume the task to continue.",
                               limit=limit, used=used, maximum=maximum)


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
