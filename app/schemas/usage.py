"""Per-task provider accounting: the budget a task may spend, the record of every provider request it made, and the
TaskUsage summary derived from those records.

Values a provider did not report stay None: usage is never guessed and prices are never invented.
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import ProviderRequestRecord, Schema
from app.schemas.providers import Capability

BUDGET_KEY = "budget"  # where a task's budget lives in Task.metadata


class TaskBudget(Schema):
    """Hard limits for one task. None means "no limit". Request and unit limits are checked before each provider
    request; limits that are only known after a request (tokens, audio seconds, cost) stop the task at the first
    request that reaches them."""

    max_llm_requests: int | None = Field(default=None, ge=0)
    max_llm_tokens: int | None = Field(default=None, ge=0)  # input + output
    max_search_requests: int | None = Field(default=None, ge=0)
    max_generated_images: int | None = Field(default=None, ge=0)
    max_searched_images: int | None = Field(default=None, ge=0)  # searched images a visual plan may ask for
    max_tts_characters: int | None = Field(default=None, ge=0)
    max_tts_seconds: float | None = Field(default=None, ge=0)
    max_generated_video_segments: int | None = Field(default=None, ge=0)
    max_generated_video_seconds: float | None = Field(default=None, ge=0)
    max_video_generation_cost_usd: float | None = Field(default=None, ge=0)  # applies to known (priced) cost
    max_cost_usd: float | None = Field(default=None, ge=0)  # applies to the cost providers or pricing report

    @classmethod
    def of(cls, metadata: dict) -> TaskBudget | None:
        raw = metadata.get(BUDGET_KEY)
        return cls.model_validate(raw) if raw else None


class TaskUsage(Schema):
    """What a task used, summed from its provider request records (plus local video rendering).

    `estimated_cost_usd` adds up only the costs that are known (provider-reported actual cost, else configured
    pricing). `cost_complete` is False as soon as one billable request had no known cost: the total is then a lower
    bound, and the unit counts are the reliable measure."""

    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    search_requests: int = 0
    image_generations: int = 0
    image_searches: int = 0
    tts_requests: int = 0
    tts_characters: int = 0
    tts_seconds: float = 0.0
    video_generations: int = 0
    video_generation_seconds: float = 0.0
    video_generation_cost_usd: float = 0.0  # known cost only (provider-reported, else configured price)
    video_render_seconds: float = 0.0
    failed_requests: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cost_complete: bool = True
    unpriced: list[str] = Field(default_factory=list)  # "<capability>:<provider>[/<model>]" without a known cost

    @property
    def llm_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_records(cls, records: list[ProviderRequestRecord], *, video_render_seconds: float = 0.0) -> TaskUsage:
        usage = cls(video_render_seconds=round(video_render_seconds, 3))
        unpriced: set[str] = set()
        for r in records:
            capability = Capability(r.capability)
            billable = r.operation in BILLABLE.get(capability, ())
            if billable:
                counter = COUNTERS[capability]
                setattr(usage, counter, getattr(usage, counter) + 1)
            if r.status == "failed":
                usage.failed_requests += 1
                continue
            usage.input_tokens += r.input_tokens or 0
            usage.output_tokens += r.output_tokens or 0
            if capability == Capability.TTS:
                usage.tts_characters += r.characters or 0
                usage.tts_seconds = round(usage.tts_seconds + (r.audio_seconds or 0.0), 3)
            if capability == Capability.VIDEO_GENERATION and billable:
                usage.video_generation_seconds = round(usage.video_generation_seconds + (r.video_seconds or 0.0), 3)
            if r.actual_cost_usd is not None:
                usage.actual_cost_usd = round(usage.actual_cost_usd + r.actual_cost_usd, 8)
            cost = r.actual_cost_usd if r.actual_cost_usd is not None else r.estimated_cost_usd
            if cost is not None:
                usage.estimated_cost_usd = round(usage.estimated_cost_usd + cost, 8)
                if capability == Capability.VIDEO_GENERATION:
                    usage.video_generation_cost_usd = round(usage.video_generation_cost_usd + cost, 8)
            elif billable and r.provider != "mock":
                unpriced.add(f"{r.capability}:{r.provider}" + (f"/{r.model}" if r.model else ""))
        usage.unpriced = sorted(unpriced)
        usage.cost_complete = not unpriced
        return usage


# The operations that spend money (and count against a budget). Listing voices or downloading an already found
# image is recorded but not billed.
BILLABLE: dict[Capability, tuple[str, ...]] = {
    Capability.LLM: ("generate",),
    Capability.SEARCH: ("search",),
    Capability.IMAGE: ("generate",),
    Capability.IMAGE_SEARCH: ("search",),
    Capability.TTS: ("synthesize",),
    Capability.VIDEO_GENERATION: ("submit",),  # polling, downloading and cancelling a job are not billed
}
COUNTERS: dict[Capability, str] = {
    Capability.LLM: "llm_requests", Capability.SEARCH: "search_requests", Capability.IMAGE: "image_generations",
    Capability.IMAGE_SEARCH: "image_searches", Capability.TTS: "tts_requests",
    Capability.VIDEO_GENERATION: "video_generations",
}
