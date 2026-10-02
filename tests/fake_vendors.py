"""In-process fakes of the vendor HTTP APIs the real adapters call, served through httpx.MockTransport.

They let the real adapters (request building, auth headers, request ids, response parsing, error mapping) run
end to end without a network or credentials. The chat fakes answer with the deterministic mock responders: the
agent is recognised by its system prompt and its input is read back from the rendered prompt, exactly as a real
model would receive it.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import struct
from collections.abc import Callable
from pathlib import Path

import httpx
from PIL import Image

from app.providers.core.http import HttpClient
from app.providers.llm.base import LLMMessage, LLMRequest
from app.providers.llm.mock import count_tokens
from app.providers.search.base import ProviderSearchRequest
from app.providers.search.mock import MockSearchProvider

INPUT_BLOCK = re.compile(r"INPUT \(JSON\):\n```json\n(.*?)\n```", re.DOTALL)
FAKE_KEY = "sk-test-0123456789abcdefghij"


def http_client(provider: str, base_url: str, handler: Callable[[httpx.Request], httpx.Response], *,
                headers: dict | None = None, timeout: float = 5.0, request_id_header: str = "X-Request-Id",
                **kwargs) -> HttpClient:
    return HttpClient(provider=provider, base_url=base_url, headers=headers or {"Authorization": f"Bearer {FAKE_KEY}"},
                      secrets=(FAKE_KEY,), timeout_seconds=timeout, transport=httpx.MockTransport(handler),
                      request_id_header=request_id_header, **kwargs)


class Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def bodies(self, path: str | None = None) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.content and (path is None or r.url.path == path)]


class FakeChatModel:
    """Answers chat requests with the mock responders. `prompts` maps a system prompt to its agent id."""

    def __init__(self, responders: dict, prompts: dict[str, str] | None = None) -> None:
        self.responders = responders
        self.prompts = prompts if prompts is not None else {}

    def answer(self, system: str, messages: list[dict], model: str) -> tuple[str, int, int]:
        agent_id = self.prompts.get(system)
        if agent_id is None:  # a direct StructuredLLM call: the first responder whose id appears in the prompt
            agent_id = next((a for a in self.responders if a in messages[0]["content"]), None)
        match = INPUT_BLOCK.search(messages[0]["content"])
        payload = json.loads(match.group(1)) if match else {}
        request = LLMRequest(model=model, system=system, agent_id=agent_id or "unknown", max_output_tokens=4000,
                             messages=[LLMMessage(role=m["role"], content=m["content"]) for m in messages],
                             attempt=sum(1 for m in messages if m["role"] == "user"), input_payload=payload)
        text = json.dumps(self.responders[agent_id](request), ensure_ascii=False)
        prompt = system + "".join(m["content"] for m in messages)
        return text, count_tokens(prompt), count_tokens(text)


def pcm_for(text: str, rate: int = 24000) -> bytes:
    """Deterministic 16-bit PCM: a short tone per word."""
    out = bytearray()
    for i, word in enumerate(text.split()):
        frames = int(rate * (len(word) + 1) / 15)
        out += b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * (k + i) / rate)))
                        for k in range(0, frames))
    return bytes(out)


def png(width: int, height: int, color=(30, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeOpenAI(Recorder):
    """Chat completions, audio speech, image generation and model lookup."""

    def __init__(self, chat: FakeChatModel | None = None) -> None:
        super().__init__()
        self.chat = chat
        self.fail: list[httpx.Response] = []  # responses returned before normal handling, in order

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            return self.fail.pop(0)
        path = request.url.path
        headers = {"x-request-id": f"vendor-{len(self.requests)}"}
        if request.method == "GET" and path.startswith("/v1/models"):
            return httpx.Response(200, json={"id": path.rsplit("/", 1)[-1], "object": "model"}, headers=headers)
        body = json.loads(request.content)
        if path == "/v1/chat/completions":
            system = body["messages"][0]["content"]
            text, tokens_in, tokens_out = self.chat.answer(system, body["messages"][1:], body["model"])
            return httpx.Response(200, headers=headers, json={
                "id": "chatcmpl-1", "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                          "prompt_tokens_details": {"cached_tokens": 0}}})
        if path == "/v1/audio/speech":
            return httpx.Response(200, content=pcm_for(body["input"]), headers={**headers,
                                                                              "content-type": "audio/pcm"})
        if path == "/v1/images/generations":
            w, h = (int(v) for v in body["size"].split("x"))
            return httpx.Response(200, headers=headers, json={
                "created": 1, "data": [{"b64_json": base64.b64encode(png(w, h)).decode()}],
                "usage": {"total_tokens": 100}})
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


class FakeAnthropic(Recorder):
    def __init__(self, chat: FakeChatModel | None = None) -> None:
        super().__init__()
        self.chat = chat
        self.fail: list[httpx.Response] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            return self.fail.pop(0)
        headers = {"request-id": f"req_{len(self.requests)}"}
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "claude-test"}]}, headers=headers)
        body = json.loads(request.content)
        text, tokens_in, tokens_out = self.chat.answer(body["system"], body["messages"], body["model"])
        return httpx.Response(200, headers=headers, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
            "usage": {"input_tokens": tokens_in - 2, "cache_read_input_tokens": 2, "output_tokens": tokens_out}})


class FakeTavily(Recorder):
    """Searches the local demo corpus, answering in Tavily's response format."""

    def __init__(self, corpus: Path) -> None:
        super().__init__()
        self._search = MockSearchProvider(corpus)
        self.fail: list[httpx.Response] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:  # async handler: used by AsyncClient
        self.requests.append(request)
        if self.fail:
            return self.fail.pop(0)
        body = json.loads(request.content)
        page = await self._search.search(ProviderSearchRequest(
            query=body["query"], max_results=body["max_results"], include_domains=body.get("include_domains", []),
            exclude_domains=body.get("exclude_domains", [])))
        results = [{"title": h.title, "url": h.url, "content": h.snippet, "raw_content": h.content, "score": h.score,
                    **({"published_date": h.published_at.isoformat()} if h.published_at else {})}
                   for h in page.hits]
        return httpx.Response(200, json={"query": body["query"], "results": results, "response_time": 0.1,
                                         "request_id": f"tvly-req-{len(self.requests)}"})
