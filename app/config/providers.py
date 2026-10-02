"""Centralised provider configuration, from environment variables (and an optional .env file).

Every capability defaults to the deterministic mock provider, so a fresh checkout runs offline with no credentials.
A real provider is used only when it is named explicitly (LLM_PROVIDER, TTS_PROVIDER, IMAGE_PROVIDER,
SEARCH_PROVIDER), and a fallback only when it is named explicitly too (<CAPABILITY>_FALLBACK_PROVIDER).

Credentials are SecretStr: they never appear in reprs, `describe()`, events or logs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.routing import ConfigError, ModelPricing, RoutingConfig, RoutingTarget
from app.schemas.common import ModelTier
from app.schemas.providers import Capability, ProviderPolicy

MOCK = "mock"

# Provider ids each capability can be configured with, and the environment variables (in order of preference)
# that hold the credential of each network provider. Mock providers need neither network nor credentials.
KNOWN_PROVIDERS: dict[Capability, dict[str, tuple[str, ...]]] = {
    Capability.LLM: {MOCK: (), "openai": ("OPENAI_API_KEY",), "anthropic": ("ANTHROPIC_API_KEY",)},
    Capability.TTS: {MOCK: (), "openai": ("TTS_API_KEY", "OPENAI_API_KEY")},
    Capability.IMAGE: {MOCK: (), "openai": ("IMAGE_API_KEY", "OPENAI_API_KEY")},
    Capability.IMAGE_SEARCH: {MOCK: ()},
    Capability.SEARCH: {MOCK: (), "tavily": ("SEARCH_API_KEY", "TAVILY_API_KEY")},
}

# Friendly role names for per-agent LLM routes (LLM_<ROLE>_PROVIDER / LLM_<ROLE>_MODEL). Any other role is taken
# as an agent id (LLM_SLIDE_PLANNER_MODEL routes the agent "slide_planner").
ROLE_ALIASES = {
    "reviewer": "content_reviewer", "planner": "curriculum_planner", "diagnostic": "knowledge_diagnostic",
    "interpreter": "request_interpreter", "evaluator": "learner_evaluation", "slides": "slide_planner",
    "audio": "audio_planner",
}
ROLE_VAR = re.compile(r"^LLM_([A-Z0-9_]+)_(PROVIDER|MODEL)$")
RESERVED_ROLES = {"FALLBACK"}


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


class ProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    teaching_agent_offline: bool = False  # TEACHING_AGENT_OFFLINE=true: only mock providers, no network at all

    # --- LLM ----------------------------------------------------------------------------------------------
    llm_provider: str | None = None  # unset: config/routing.toml decides (mock by default)
    llm_model: str | None = None
    llm_fallback_provider: str | None = None
    llm_fallback_model: str | None = None
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    llm_input_price_per_mtok: float | None = Field(default=None, ge=0)  # USD; without it cost is "unknown"
    llm_output_price_per_mtok: float | None = Field(default=None, ge=0)
    llm_native_structured_output: bool = True  # use the provider's JSON-schema output mode when the schema fits
    # Per-agent routes: agent id -> (provider, model). Read from LLM_<ROLE>_PROVIDER / LLM_<ROLE>_MODEL when not
    # given explicitly (pass {} to ignore the environment).
    llm_routes: dict[str, tuple[str | None, str | None]] | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_organization: str | None = None
    openai_max_tokens_param: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str = "https://api.anthropic.com"

    # --- TTS ----------------------------------------------------------------------------------------------
    tts_provider: str = MOCK
    tts_fallback_provider: str | None = None
    tts_api_key: SecretStr | None = None
    tts_model: str | None = None
    tts_base_url: str | None = None
    tts_languages: str = "en-US,en-GB,es-ES,es-MX,fr-FR,de-DE,zh-CN"  # languages a multilingual voice is offered in

    # --- Image generation and image search -----------------------------------------------------------------
    image_provider: str = MOCK
    image_fallback_provider: str | None = None
    image_api_key: SecretStr | None = None
    image_model: str | None = None
    image_base_url: str | None = None
    image_search_provider: str = MOCK

    # --- Web search ---------------------------------------------------------------------------------------
    search_provider: str = MOCK
    search_fallback_provider: str | None = None
    search_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    search_base_url: str | None = None
    search_depth: Literal["basic", "advanced"] = "basic"
    search_include_raw_content: bool = True  # full page text, so evidence quotes can be checked against it
    search_include_domains: str | None = None  # comma-separated; results are restricted to these domains
    search_exclude_domains: str | None = None

    # --- Call policy: timeouts, bounded retries, local rate limits ------------------------------------------
    provider_backoff_seconds: float = Field(default=0.5, ge=0, le=60)
    provider_backoff_multiplier: float = Field(default=2.0, ge=1, le=10)
    provider_max_backoff_seconds: float = Field(default=8.0, ge=0, le=300)
    provider_max_request_bytes: int = Field(default=4_000_000, ge=1024)
    provider_max_response_bytes: int = Field(default=32_000_000, ge=1024)
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    llm_requests_per_minute: int | None = Field(default=None, ge=1)
    llm_max_concurrency: int | None = Field(default=None, ge=1)
    tts_timeout_seconds: float = Field(default=90.0, gt=0, le=3600)
    tts_max_attempts: int = Field(default=3, ge=1, le=10)
    tts_requests_per_minute: int | None = Field(default=None, ge=1)
    tts_max_concurrency: int | None = Field(default=None, ge=1)
    image_timeout_seconds: float = Field(default=110.0, gt=0, le=3600)
    image_max_attempts: int = Field(default=2, ge=1, le=10)
    image_requests_per_minute: int | None = Field(default=None, ge=1)
    image_max_concurrency: int | None = Field(default=None, ge=1)
    image_search_timeout_seconds: float = Field(default=15.0, gt=0, le=3600)
    image_search_max_attempts: int = Field(default=3, ge=1, le=10)
    search_timeout_seconds: float = Field(default=15.0, gt=0, le=3600)
    search_max_attempts: int = Field(default=3, ge=1, le=10)
    search_requests_per_minute: int | None = Field(default=None, ge=1)
    search_max_concurrency: int | None = Field(default=None, ge=1)

    @field_validator("*", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object, info: ValidationInfo) -> object:
        """An empty variable (LLM_PROVIDER=) means "not set": the field's default."""
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].get_default(call_default_factory=True)
        return value

    def model_post_init(self, _context) -> None:
        if self.llm_routes is None:
            self.llm_routes = parse_llm_role_routes(os.environ)
        # the "default" role is LLM_PROVIDER / LLM_MODEL under another name
        provider, model = self.llm_routes.pop("default", (None, None))
        self.llm_provider = self.llm_provider or provider
        self.llm_model = self.llm_model or model

    # --- Derived views --------------------------------------------------------------------------------------

    @property
    def offline(self) -> bool:
        return self.teaching_agent_offline

    def policies(self) -> dict[Capability, ProviderPolicy]:
        backoff = {"backoff_seconds": self.provider_backoff_seconds,
                   "backoff_multiplier": self.provider_backoff_multiplier,
                   "max_backoff_seconds": self.provider_max_backoff_seconds}
        return {
            Capability.LLM: ProviderPolicy(timeout_seconds=self.llm_timeout_seconds, max_attempts=self.llm_max_attempts,
                                           requests_per_minute=self.llm_requests_per_minute,
                                           max_concurrency=self.llm_max_concurrency, **backoff),
            Capability.TTS: ProviderPolicy(timeout_seconds=self.tts_timeout_seconds, max_attempts=self.tts_max_attempts,
                                           requests_per_minute=self.tts_requests_per_minute,
                                           max_concurrency=self.tts_max_concurrency, **backoff),
            Capability.IMAGE: ProviderPolicy(timeout_seconds=self.image_timeout_seconds,
                                             max_attempts=self.image_max_attempts,
                                             requests_per_minute=self.image_requests_per_minute,
                                             max_concurrency=self.image_max_concurrency, **backoff),
            Capability.IMAGE_SEARCH: ProviderPolicy(timeout_seconds=self.image_search_timeout_seconds,
                                                    max_attempts=self.image_search_max_attempts, **backoff),
            Capability.SEARCH: ProviderPolicy(timeout_seconds=self.search_timeout_seconds,
                                              max_attempts=self.search_max_attempts,
                                              requests_per_minute=self.search_requests_per_minute,
                                              max_concurrency=self.search_max_concurrency, **backoff),
        }

    def primary(self, capability: Capability) -> str | None:
        """The configured provider id of a capability. For the LLM, None means "as config/routing.toml says"."""
        return {Capability.LLM: self.llm_provider, Capability.TTS: self.tts_provider,
                Capability.IMAGE: self.image_provider, Capability.IMAGE_SEARCH: self.image_search_provider,
                Capability.SEARCH: self.search_provider}[capability]

    def fallback(self, capability: Capability) -> str | None:
        return {Capability.LLM: self.llm_fallback_provider, Capability.TTS: self.tts_fallback_provider,
                Capability.IMAGE: self.image_fallback_provider, Capability.IMAGE_SEARCH: None,
                Capability.SEARCH: self.search_fallback_provider}[capability]

    def model(self, capability: Capability) -> str | None:
        return {Capability.LLM: self.llm_model, Capability.TTS: self.tts_model, Capability.IMAGE: self.image_model,
                Capability.IMAGE_SEARCH: None, Capability.SEARCH: None}[capability]

    def chain(self, capability: Capability) -> list[str]:
        """Primary then the explicitly configured fallback, for every capability but the LLM (routed by tier)."""
        return [p for p in (self.primary(capability), self.fallback(capability)) if p]

    def _secret_fields(self) -> dict[str, SecretStr | None]:
        return {"OPENAI_API_KEY": self.openai_api_key, "ANTHROPIC_API_KEY": self.anthropic_api_key,
                "TTS_API_KEY": self.tts_api_key, "IMAGE_API_KEY": self.image_api_key,
                "SEARCH_API_KEY": self.search_api_key, "TAVILY_API_KEY": self.tavily_api_key}

    def secret_values(self) -> list[str]:
        return [s.get_secret_value() for s in self._secret_fields().values() if s and s.get_secret_value()]

    def credential(self, capability: Capability, provider: str) -> tuple[str | None, str | None]:
        """(secret value, env var it came from) for a network provider, or (None, None) when none is set."""
        fields = self._secret_fields()
        for var in KNOWN_PROVIDERS.get(capability, {}).get(provider, ()):
            secret = fields.get(var)
            if secret is not None and secret.get_secret_value():
                return secret.get_secret_value(), var
        return None, None

    def include_domains(self) -> list[str]:
        return _csv(self.search_include_domains)

    def exclude_domains(self) -> list[str]:
        return _csv(self.search_exclude_domains)

    def languages(self) -> list[str]:
        return _csv(self.tts_languages)

    # --- Validation ----------------------------------------------------------------------------------------

    def problems(self) -> list[str]:
        """Everything wrong with the provider configuration, as messages naming the variables to set. Secret values
        never appear in them."""
        problems: list[str] = []
        configured: list[tuple[Capability, str, str]] = []  # (capability, provider, variable that selected it)
        for cap in Capability:
            prefix = cap.name  # LLM, TTS, IMAGE, IMAGE_SEARCH, SEARCH
            for kind, provider in (("PROVIDER", self.primary(cap)), ("FALLBACK_PROVIDER", self.fallback(cap))):
                if provider:
                    configured.append((cap, provider, f"{prefix}_{kind}"))
            if self.fallback(cap) and self.fallback(cap) == self.primary(cap):
                problems.append(f"{prefix}_FALLBACK_PROVIDER repeats {prefix}_PROVIDER ('{self.primary(cap)}')")
        for agent_id, (provider, _model) in (self.llm_routes or {}).items():
            if provider:
                configured.append((Capability.LLM, provider, f"the LLM route of agent '{agent_id}'"))

        for cap, provider, var in configured:
            known = KNOWN_PROVIDERS[cap]
            if provider not in known:
                problems.append(f"{var}='{provider}' is not a known {cap.value} provider (known: {sorted(known)})")
                continue
            if provider == MOCK:
                continue
            if self.offline:
                problems.append(f"{var}='{provider}' needs the network, but TEACHING_AGENT_OFFLINE=true allows only "
                                "mock providers")
            if self.credential(cap, provider)[0] is None:
                problems.append(f"{var}='{provider}' needs a credential: set {' or '.join(known[provider])}")

        if self.llm_provider and self.llm_provider != MOCK and not self.llm_model:
            problems.append(f"LLM_PROVIDER='{self.llm_provider}' needs LLM_MODEL")
        if bool(self.llm_fallback_provider) != bool(self.llm_fallback_model):
            problems.append("LLM_FALLBACK_PROVIDER and LLM_FALLBACK_MODEL must be set together")
        if self.llm_fallback_provider and not self.llm_provider:
            problems.append("LLM_FALLBACK_PROVIDER needs LLM_PROVIDER (or define fallbacks in config/routing.toml)")
        if (self.llm_input_price_per_mtok is None) != (self.llm_output_price_per_mtok is None):
            problems.append("LLM_INPUT_PRICE_PER_MTOK and LLM_OUTPUT_PRICE_PER_MTOK must be set together")
        for agent_id, (provider, model) in (self.llm_routes or {}).items():
            if not provider or not model:
                problems.append(f"the LLM route of agent '{agent_id}' needs both LLM_<ROLE>_PROVIDER and "
                                "LLM_<ROLE>_MODEL")
        return problems

    def validate_startup(self) -> None:
        problems = self.problems()
        if problems:
            raise ConfigError("provider configuration is invalid:\n  - " + "\n  - ".join(problems))

    def describe(self) -> dict:
        """The configuration as safe-to-print data: providers, models, fallbacks and whether each credential is
        set. Never a secret value."""
        out: dict = {"offline": self.offline, "capabilities": {}}
        for cap in Capability:
            entry: dict = {"provider": self.primary(cap) or "(config/routing.toml)", "model": self.model(cap),
                           "fallback": self.fallback(cap)}
            creds = {}
            for provider in filter(None, (self.primary(cap), self.fallback(cap))):
                if provider in KNOWN_PROVIDERS[cap] and provider != MOCK:
                    _, var = self.credential(cap, provider)
                    creds[provider] = f"set ({var})" if var else "missing"
            if creds:
                entry["credentials"] = creds
            out["capabilities"][cap.value] = entry
        return out


def parse_llm_role_routes(environ: Mapping[str, str]) -> dict[str, tuple[str | None, str | None]]:
    """Per-role LLM routes from LLM_<ROLE>_PROVIDER / LLM_<ROLE>_MODEL, keyed by agent id."""
    found: dict[str, dict[str, str]] = {}
    for key, value in environ.items():
        match = ROLE_VAR.match(key.upper())
        if match and match.group(1) not in RESERVED_ROLES and value:
            found.setdefault(match.group(1), {})[match.group(2)] = value
    routes = {}
    for role, parts in found.items():
        agent_id = ROLE_ALIASES.get(role.lower(), role.lower())
        routes[agent_id] = (parts.get("PROVIDER"), parts.get("MODEL"))
    return routes


def apply_llm_overrides(routing: RoutingConfig, settings: ProviderSettings) -> RoutingConfig:
    """The routing file with the environment's LLM choices applied: LLM_PROVIDER/LLM_MODEL (plus an explicit
    LLM_FALLBACK_*) replace every tier's chain; LLM_<ROLE>_* give one agent its own route."""
    config = routing.model_copy(deep=True)
    if settings.llm_temperature is not None:
        config.temperature = settings.llm_temperature
    env_models: list[str] = []
    if settings.llm_provider and settings.llm_model:
        chain = [RoutingTarget(provider=settings.llm_provider, model=settings.llm_model)]
        if settings.llm_fallback_provider and settings.llm_fallback_model:
            chain.append(RoutingTarget(provider=settings.llm_fallback_provider, model=settings.llm_fallback_model))
        config.tiers = {tier: list(chain) for tier in ModelTier}
        env_models += [t.model for t in chain]
    for agent_id, (provider, model) in (settings.llm_routes or {}).items():
        if provider and model:
            config.routes[agent_id] = [RoutingTarget(provider=provider, model=model)]
            env_models.append(model)
    if settings.llm_input_price_per_mtok is not None and settings.llm_model:
        config.pricing[settings.llm_model] = ModelPricing(input_per_mtok=settings.llm_input_price_per_mtok,
                                                          output_per_mtok=settings.llm_output_price_per_mtok or 0.0)
    config.unpriced_models = sorted({*config.unpriced_models, *(m for m in env_models if m not in config.pricing)})
    return config
