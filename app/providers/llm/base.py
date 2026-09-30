"""Provider-independent LLM contract. Concrete providers adapt a vendor SDK/API to this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import Field

from app.schemas.common import Schema, TokenUsage


class LLMMessage(Schema):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(Schema):
    model: str
    system: str
    messages: list[LLMMessage] = Field(min_length=1)
    response_schema: dict | None = None
    max_output_tokens: int = Field(ge=1)
    temperature: float = 0.0
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


class ProviderError(Exception):
    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
