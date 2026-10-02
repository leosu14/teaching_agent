"""LLM adapter for the Anthropic Messages API (Claude). Plain HTTP: no vendor SDK.

Structured output: a schema that fits Anthropic's structured-output subset is sent as
`output_config.format = {type: json_schema, schema}`; otherwise the schema stays in the prompt and the caller
validates the JSON text. Temperature is sent only when configured (current Claude models reject it).
"""

from __future__ import annotations

from typing import ClassVar

from app.providers.core.errors import ProviderResponseError
from app.providers.core.http import HttpClient
from app.providers.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.providers.llm.structured import native_schema
from app.schemas.common import TokenUsage

API_VERSION = "2023-06-01"


class AnthropicLLMProvider(LLMProvider):
    name = "anthropic"
    requires_network: ClassVar[bool] = True
    structured_output: ClassVar[bool] = True

    def __init__(self, http: HttpClient, *, native_structured_output: bool = True) -> None:
        self._http = http
        self._native = native_structured_output

    def configuration(self) -> dict:
        return {"base_url": self._http.base_url, "api_version": API_VERSION,
                "native_structured_output": self._native, "timeout_seconds": self._http.timeout_seconds}

    async def probe(self) -> str:
        await self._http.request("GET", "/v1/models", params={"limit": "1"})  # authenticated, no tokens spent
        return "network"

    def build_body(self, request: LLMRequest) -> tuple[dict, bool]:
        body: dict = {"model": request.model, "max_tokens": request.max_output_tokens, "system": request.system,
                      "messages": [{"role": m.role, "content": m.content} for m in request.messages]}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        structured = False
        if request.response_schema is not None and self._native:
            schema = native_schema(request.response_schema)
            if schema is not None:
                body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
                structured = True
        return body, structured

    async def generate(self, request: LLMRequest) -> LLMResponse:
        body, structured = self.build_body(request)
        response = await self._http.request("POST", "/v1/messages", json_body=body)
        data = response.json()
        stop_reason = data.get("stop_reason") or "end_turn"
        if stop_reason == "refusal":
            category = (data.get("stop_details") or {}).get("category")
            raise ProviderResponseError(f"{self.name}: the model declined the request (category: {category})",
                                        provider=self.name)
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderResponseError(f"{self.name}: response has no content", provider=self.name)
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        usage = data.get("usage") or {}
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_write = usage.get("cache_creation_input_tokens") or 0
        # Anthropic reports cached input separately; TokenUsage.input_tokens counts all input, cached included.
        return LLMResponse(
            text=text, provider=self.name, model=data.get("model") or request.model,
            usage=TokenUsage(input_tokens=(usage.get("input_tokens") or 0) + cache_read + cache_write,
                             output_tokens=usage.get("output_tokens") or 0, cached_input_tokens=cache_read),
            stop_reason=stop_reason, structured=structured, vendor_request_id=response.vendor_request_id,
        )
