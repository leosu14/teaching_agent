"""Provider-independent LLM contract. Concrete providers adapt a vendor SDK/API to this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from pydantic import Field

from app.providers.core.base import Provider
from app.schemas.common import Schema, TokenUsage
from app.schemas.providers import Capability


class LLMMessage(Schema):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(Schema):
    model: str
    system: str
    messages: list[LLMMessage] = Field(min_length=1)
    response_schema: dict | None = None
    max_output_tokens: int = Field(ge=1)
    temperature: float | None = None  # None: the provider's default; adapters send it only where it is supported
    agent_id: str
    attempt: int = 1
    # The structured input the prompt was rendered from. Real providers send only `messages`;
    # the deterministic mock provider reads it so it does not have to parse prose.
    input_payload: dict = Field(default_factory=dict)


class LLMResponse(Schema):
    text: str
    provider: str
    model: str
    usage: TokenUsage
    stop_reason: str = "end_turn"
    structured: bool = False  # True when the provider enforced `response_schema` natively
    vendor_request_id: str | None = None


class LLMProvider(Provider, ABC):
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.LLM})
    # Whether the provider can constrain output to a JSON schema itself. Either way the caller (StructuredLLM)
    # parses and validates the text, so a provider without it still works through the JSON-in-prompt fallback.
    structured_output: ClassVar[bool] = False

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
