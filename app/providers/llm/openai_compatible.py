"""LLM adapter for the OpenAI Chat Completions API and servers compatible with it (Azure OpenAI's v1 endpoint, vLLM,
Ollama, LM Studio, ...). Plain HTTP: no vendor SDK.

Structured output: when the request carries a schema that fits OpenAI's strict JSON-schema mode, it is sent as
`response_format: json_schema` (strict); otherwise JSON mode (`json_object`) is requested and the caller validates.
"""

from __future__ import annotations

from typing import ClassVar

from app.providers.core.errors import ProviderResponseError
from app.providers.core.http import HttpClient
from app.providers.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.providers.llm.structured import native_schema
from app.schemas.common import TokenUsage

FINISH_REASONS = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal", "tool_calls": "tool_use"}


class OpenAICompatibleLLMProvider(LLMProvider):
    requires_network: ClassVar[bool] = True
    structured_output: ClassVar[bool] = True

    def __init__(self, http: HttpClient, *, name: str = "openai", native_structured_output: bool = True,
                 max_tokens_param: str = "max_completion_tokens") -> None:
        self.name = name
        self._http = http
        self._native = native_structured_output
        self._max_tokens_param = max_tokens_param

    def configuration(self) -> dict:
        return {"base_url": self._http.base_url, "native_structured_output": self._native,
                "max_tokens_param": self._max_tokens_param, "timeout_seconds": self._http.timeout_seconds}

    async def probe(self) -> str:
        await self._http.request("GET", "/models")  # lists models: authenticated, no tokens spent
        return "network"

    def build_body(self, request: LLMRequest) -> tuple[dict, bool]:
        messages = [{"role": "system", "content": request.system}]
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        body: dict = {"model": request.model, "messages": messages, self._max_tokens_param: request.max_output_tokens}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        structured = False
        if request.response_schema is not None and self._native:
            strict = native_schema(request.response_schema, all_required=True)
            if strict is not None:
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "output", "schema": strict, "strict": True}}
                structured = True
            else:  # the schema has free-form parts: ask for a JSON object, validated by the caller
                body["response_format"] = {"type": "json_object"}
        return body, structured

    async def generate(self, request: LLMRequest) -> LLMResponse:
        body, structured = self.build_body(request)
        response = await self._http.request("POST", "/chat/completions", json_body=body)
        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError(f"{self.name}: response has no choices", provider=self.name) from exc
        if message.get("refusal"):
            raise ProviderResponseError(f"{self.name}: the model refused: {message['refusal'][:200]}",
                                        provider=self.name)
        text = message.get("content")
        if not isinstance(text, str):
            raise ProviderResponseError(f"{self.name}: response has no text content", provider=self.name)
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return LLMResponse(
            text=text, provider=self.name, model=data.get("model") or request.model,
            usage=TokenUsage(input_tokens=usage.get("prompt_tokens") or 0,
                             output_tokens=usage.get("completion_tokens") or 0, cached_input_tokens=cached),
            stop_reason=FINISH_REASONS.get(choice.get("finish_reason"), choice.get("finish_reason") or "end_turn"),
            structured=structured, vendor_request_id=response.vendor_request_id,
        )
