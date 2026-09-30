from __future__ import annotations

import pytest

from app.agents.base import Agent, AgentContext, AgentOutputError, AgentSpec, OutputRejected
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.router import ModelRouter
from app.schemas.common import ModelTier, Schema
from app.tools.base import ToolPermissionError
from app.tools.manager import ToolManager
from app.tools.registry import ToolRegistry
from tests.unit.helpers import routing, scope
from tests.unit.test_tools import Echo


class In(Schema):
    n: int


class Out(Schema):
    doubled: int


class Doubler(Agent[In, Out]):
    spec = AgentSpec(id="doubler", name="Doubler", description="doubles", input_model=In, output_model=Out,
                     tier=ModelTier.CHEAP, validation_retries=2)

    def check(self, output: Out, source: In) -> None:
        if output.doubled != source.n * 2:
            raise OutputRejected("doubled must be exactly twice n")

    async def run(self, data, ctx):
        if data.n < 0:
            await self.use_tool("search.web", {"query": "x"}, ctx)
        return await self.generate(data, ctx, source=data)


def make(llm: MockLLMProvider):
    router = ModelRouter(routing(("mock", "m1")), {"mock": llm})
    sc, events = scope()
    return Doubler(system_prompt="Double the number."), AgentContext(router=router, tools=ToolManager(ToolRegistry()), scope=sc), events


async def test_valid_output_first_time() -> None:
    llm = MockLLMProvider({"doubler": lambda r: {"doubled": r.input_payload["n"] * 2}})
    agent, ctx, events = make(llm)
    assert await agent.execute({"n": 4}, ctx) == Out(doubled=8)
    assert [e.type for e in events] == ["agent.started", "llm.call", "agent.finished"]
    assert llm.requests[0].response_schema == Out.model_json_schema()


async def test_malformed_and_semantically_wrong_output_are_corrected() -> None:
    llm = MockLLMProvider({"doubler": lambda r: {"doubled": r.input_payload["n"] * 2}})
    llm.inject("doubler", "```json\n{\"doubled\": \"many\"}\n```", '{"doubled": 5}')
    agent, ctx, events = make(llm)
    assert await agent.execute({"n": 4}, ctx) == Out(doubled=8)
    assert [e.type for e in events].count("agent.validation_failed") == 2
    assert "twice" in llm.requests[2].messages[-1].content  # semantic error fed back to the model


async def test_gives_up_after_validation_budget() -> None:
    llm = MockLLMProvider({})
    llm.inject("doubler", *["nope"] * 10)
    agent, ctx, events = make(llm)
    with pytest.raises(AgentOutputError):
        await agent.execute({"n": 1}, ctx)
    assert llm.calls["doubler"] == 3 and events[-1].type == "agent.failed"


async def test_input_is_validated_and_tools_are_allow_listed() -> None:
    agent, ctx, _ = make(MockLLMProvider({}))
    with pytest.raises(ValueError):
        await agent.execute({"n": "x"}, ctx)
    ctx.tools.registry.register(Echo())
    with pytest.raises(ToolPermissionError):
        await agent.execute({"n": -1}, ctx)
