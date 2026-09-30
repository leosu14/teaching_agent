"""Deterministic mock LLM provider.

It returns text exactly like a real provider does, so agent output goes through the same
JSON parsing and schema validation path. Faults (malformed text, provider errors) can be
injected per agent to exercise retries and fallback.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Callable

from app.providers.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.schemas.common import TokenUsage

Responder = Callable[[LLMRequest], dict]


def count_tokens(text: str) -> int:
    """Deterministic token estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4))


class MockLLMProvider(LLMProvider):
    def __init__(self, responders: dict[str, Responder], name: str = "mock") -> None:
        self.name = name
        self._responders = responders
        self._faults: dict[str, deque[str | Exception]] = {}
        self.calls: Counter[str] = Counter()
        self.requests: list[LLMRequest] = []

    def inject(self, agent_id: str, *faults: str | Exception) -> None:
        """Queue raw text or exceptions returned before the responder is used for `agent_id`."""
        self._faults.setdefault(agent_id, deque()).extend(faults)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls[request.agent_id] += 1
        self.requests.append(request)
        queue = self._faults.get(request.agent_id)
        if queue:
            fault = queue.popleft()
            if isinstance(fault, Exception):
                raise fault
            text = fault
        else:
            responder = self._responders.get(request.agent_id)
            if responder is None:
                raise KeyError(f"mock provider has no responder for agent '{request.agent_id}'")
            text = json.dumps(responder(request), ensure_ascii=False)
        prompt_text = request.system + "".join(m.content for m in request.messages)
        usage = TokenUsage(input_tokens=count_tokens(prompt_text), output_tokens=count_tokens(text))
        return LLMResponse(text=text, provider=self.name, model=request.model, usage=usage)
