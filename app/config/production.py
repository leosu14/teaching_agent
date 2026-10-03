"""Production run configuration: the per-task budget and the production readiness rules.

Budgets come from unprefixed environment variables (like the provider variables) and are stored in each production
task's metadata, so they are visible with the task and enforced by the provider layer on every request.

Readiness: a production run needs TEACHING_AGENT_MODE=production and, for every capability its workflow uses, a
real provider selected explicitly with its credential. Capabilities the workflow does not use are not required.
"""

from __future__ import annotations

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.providers import KNOWN_PROVIDERS, MOCK, ProviderSettings
from app.schemas.providers import Capability
from app.schemas.usage import TaskBudget

# The variable that selects each capability's provider, for messages.
PROVIDER_VARS = {Capability.LLM: "LLM_PROVIDER", Capability.TTS: "TTS_PROVIDER", Capability.IMAGE: "IMAGE_PROVIDER",
                 Capability.IMAGE_SEARCH: "IMAGE_SEARCH_PROVIDER", Capability.SEARCH: "SEARCH_PROVIDER",
                 Capability.VIDEO_GENERATION: "VIDEO_GENERATION_PROVIDER"}


class ProductionSettings(BaseSettings):
    """Limits of one production task. Every limit can be raised or lowered; `max_cost_usd` has no default because
    vendor prices are not built in (configure LLM pricing with LLM_INPUT/OUTPUT_PRICE_PER_MTOK to make it bite)."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    max_llm_requests: int = Field(default=80, ge=1)
    max_llm_tokens: int = Field(default=600_000, ge=1)
    max_search_requests: int = Field(default=20, ge=0)
    max_generated_images: int = Field(default=4, ge=0)
    # No real image search adapter exists yet (searched images come from the local catalog), so production visual
    # plans use generated visuals only unless this is raised explicitly.
    max_searched_images: int = Field(default=0, ge=0)
    max_tts_characters: int = Field(default=30_000, ge=1)
    max_tts_seconds: float = Field(default=1800.0, gt=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    # Generated video segments (optional; only lessons that ask for them). They apply to every lesson, not only to
    # production tasks: the video strategy plans within them, and production tasks also enforce them per request.
    max_generated_video_segments: int = Field(default=2, ge=0, le=20)
    max_generated_video_seconds: float = Field(default=20.0, ge=0, le=600)
    max_video_generation_cost_usd: float | None = Field(default=None, ge=0)  # None: no cost limit (counts still do)
    production_health_timeout_seconds: float = Field(default=20.0, gt=0, le=300)

    @field_validator("*", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].get_default(call_default_factory=True)
        return value

    def budget(self) -> TaskBudget:
        return TaskBudget(
            max_llm_requests=self.max_llm_requests, max_llm_tokens=self.max_llm_tokens,
            max_search_requests=self.max_search_requests, max_generated_images=self.max_generated_images,
            max_searched_images=self.max_searched_images, max_tts_characters=self.max_tts_characters,
            max_tts_seconds=self.max_tts_seconds, max_cost_usd=self.max_cost_usd,
            max_generated_video_segments=self.max_generated_video_segments,
            max_generated_video_seconds=self.max_generated_video_seconds,
            max_video_generation_cost_usd=self.max_video_generation_cost_usd,
        )


def required_capabilities(workflow_capabilities: frozenset[Capability], budget: TaskBudget,
                          *, generated_video: bool = False) -> set[Capability]:
    """The provider capabilities a run needs: what the workflow uses, minus what its budget switches off. Video
    generation is only needed by a lesson that asks for generated segments, with a budget that allows one."""
    needed = set(workflow_capabilities)
    if not generated_video or not budget.max_generated_video_segments or not budget.max_generated_video_seconds:
        needed.discard(Capability.VIDEO_GENERATION)
    if budget.max_generated_images == 0:
        needed.discard(Capability.IMAGE)
    if budget.max_searched_images == 0:
        needed.discard(Capability.IMAGE_SEARCH)
    return needed


def production_readiness(providers: ProviderSettings, required: set[Capability]) -> tuple[list[str], list[str]]:
    """(problems, warnings) of running a production task that needs `required`. Problems name the variable to set;
    secret values never appear."""
    problems = list(providers.problems())
    warnings: list[str] = []
    if providers.mode != "production":
        problems.insert(0, "production mode is not configured: set TEACHING_AGENT_MODE=production (the mode is "
                           f"'{providers.mode}', which runs mock providers only)")
    for capability in sorted(required, key=lambda c: c.value):
        var = PROVIDER_VARS[capability]
        real = sorted(p for p in KNOWN_PROVIDERS[capability] if p != MOCK)
        provider = providers.primary(capability)
        if capability == Capability.IMAGE_SEARCH:
            # No real image search adapter exists yet: the local catalog is the only source, said out loud.
            warnings.append("searched images come from the local image catalog (TA_CORPUS_DIR/image_catalog.json): "
                            "there is no real image search adapter yet; set MAX_SEARCHED_IMAGES=0 to plan generated "
                            "visuals only")
            continue
        if not provider or (provider == MOCK and var.lower() not in providers.model_fields_set):
            problems.append(f"the workflow needs a provider for {capability.value}: set {var} (one of {real})"
                            + (" and LLM_MODEL" if capability == Capability.LLM else ""))
        elif provider == MOCK:
            problems.append(f"{var}='mock' is a mock: production mode needs a real {capability.value} provider "
                            f"(one of {real})")
    return problems, warnings
