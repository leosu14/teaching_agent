from __future__ import annotations

import asyncio

import pytest

from app.schemas.common import RetryPolicy, Schema
from app.tools.base import Tool, ToolCaller, ToolError, ToolNotFound, ToolPermissionError, ToolTransientError
from app.tools.manager import ToolManager
from app.tools.registry import ToolRegistry
from tests.unit.helpers import scope


class Q(Schema):
    query: str


class R(Schema):
    result: str


class Echo(Tool[Q, R]):
    name = "search.web"
    description = "echo"
    input_model = Q
    output_model = R
    permissions = frozenset({"network"})

    async def run(self, data, scope):
        return R(result=data.query)


class Flaky(Tool[Q, R]):
    name = "flaky"
    description = "fails twice"
    input_model = Q
    output_model = R
    retry = RetryPolicy(max_attempts=3)
    timeout_seconds = 0.05

    def __init__(self, mode: str) -> None:
        self.mode, self.calls = mode, 0

    async def run(self, data, scope):
        self.calls += 1
        if self.calls < 3:
            if self.mode == "slow":
                await asyncio.sleep(1)
            raise ToolTransientError("blip")
        return R(result="ok")


class Wrong(Tool[Q, R]):
    name = "wrong"
    description = "bad output"
    input_model = Q
    output_model = R

    async def run(self, data, scope):
        return Q(query="not an R")


def manager(*tools: Tool) -> ToolManager:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return ToolManager(registry)


def caller(*tools: str, perms: set[str] = frozenset()) -> ToolCaller:
    return ToolCaller(caller_id="test", allowed_tools=frozenset(tools), permissions=frozenset(perms))


async def test_call_validates_and_emits_events() -> None:
    sc, events = scope()
    out = await manager(Echo()).call(caller("search.web", perms={"network"}), "search.web", {"query": "hi"}, sc)
    assert out == R(result="hi")
    assert [e.type for e in events] == ["tool.started", "tool.finished"] and events[0].tool == "search.web"


async def test_permissions_and_allow_list_are_enforced() -> None:
    m = manager(Echo())
    with pytest.raises(ToolPermissionError, match="not allowed"):
        await m.call(caller(), "search.web", {"query": "x"}, scope()[0])
    with pytest.raises(ToolPermissionError, match="lacks permissions"):
        await m.call(caller("search.web"), "search.web", {"query": "x"}, scope()[0])
    with pytest.raises(ToolNotFound):
        await m.call(caller("nope"), "nope", {}, scope()[0])
    with pytest.raises(ToolError, match="invalid input"):
        await m.call(caller("search.web", perms={"network"}), "search.web", {"q": 1}, scope()[0])


@pytest.mark.parametrize("mode", ["error", "slow"])
async def test_transient_failures_and_timeouts_are_retried(mode) -> None:
    tool = Flaky(mode)
    sc, events = scope()
    assert await manager(tool).call(caller("flaky"), "flaky", {"query": "x"}, sc) == R(result="ok")
    assert tool.calls == 3
    assert [e.type for e in events].count("tool.failed") == 2


async def test_wrong_output_type_is_rejected() -> None:
    with pytest.raises(ToolError, match="expected R"):
        await manager(Wrong()).call(caller("wrong"), "wrong", {"query": "x"}, scope()[0])


def test_registry_rejects_duplicates_and_describes_tools() -> None:
    registry = ToolRegistry()
    registry.register(Echo())
    with pytest.raises(ValueError):
        registry.register(Echo())
    info = registry.describe()[0]
    assert info.name == "search.web" and info.permissions == ["network"] and "query" in str(info.input_schema)
