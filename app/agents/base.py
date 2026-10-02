"""Agent runtime contract.

An agent decides WHAT to do: it gathers context through tools (ToolManager) and asks a model
(through StructuredLLM and the ModelRouter) for a structured result, which is always parsed and
schema-validated. A result that fails validation is sent back to the model with the error, up to
`validation_retries` times. Agents never see a provider or vendor: only the router and StructuredLLM.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from app.observability.scope import ExecutionScope
from app.providers.llm.router import AllProvidersFailed, ModelRouter
from app.providers.llm.structured import StructuredLLM, StructuredOutputError
from app.schemas.catalog import AgentInfo
from app.schemas.common import ModelTier, RetryPolicy
from app.schemas.events import EventType
from app.tools.base import ToolCaller
from app.tools.manager import ToolManager
from app.utils.retry import retry_async

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)


class AgentError(Exception):
    pass


class AgentOutputError(AgentError):
    """The model never produced output that passed validation."""


class OutputRejected(ValueError):
    """Raised by `Agent.check` when output is schema-valid but semantically wrong."""


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    tier: ModelTier
    tools: tuple[str, ...] = ()
    permissions: frozenset[str] = field(default_factory=frozenset)
    timeout_seconds: float = 120.0
    llm_timeout_seconds: float = 60.0
    max_output_tokens: int = 4000
    validation_retries: int = 2
    retry: RetryPolicy = field(default_factory=lambda: RetryPolicy(max_attempts=2))




@dataclass(frozen=True)
class AgentContext:
    router: ModelRouter
    tools: ToolManager
    scope: ExecutionScope


class Agent(ABC, Generic[I, O]):
    spec: ClassVar[AgentSpec]
    instructions: ClassVar[str] = "Produce the output for the input below."

    def __init__(self, system_prompt: str | None = None) -> None:
        if system_prompt is None:  # default: prompt.md next to the agent's module
            system_prompt = Path(inspect.getfile(type(self))).with_name("prompt.md").read_text(encoding="utf-8")
        self.system_prompt = system_prompt
        self.caller = ToolCaller(caller_id=f"agent:{self.spec.id}", allowed_tools=frozenset(self.spec.tools),
                                 permissions=self.spec.permissions)

    def info(self) -> AgentInfo:
        s = self.spec
        return AgentInfo(id=s.id, name=s.name, description=s.description, tier=s.tier, tools=list(s.tools),
                         permissions=sorted(s.permissions), timeout_seconds=s.timeout_seconds,
                         validation_retries=s.validation_retries,
                         input_schema=s.input_model.model_json_schema(),
                         output_schema=s.output_model.model_json_schema())

    # --- public entry point -----------------------------------------------------------------

    async def execute(self, payload: BaseModel | dict, ctx: AgentContext) -> O:
        data = self.spec.input_model.model_validate(
            payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        )
        scope = ctx.scope.for_agent(self.spec.id)
        ctx = AgentContext(router=ctx.router, tools=ctx.tools, scope=scope)
        started = time.perf_counter()
        scope.emit(EventType.AGENT_STARTED, name=self.spec.name)

        async def attempt(n: int) -> O:
            return await asyncio.wait_for(self.run(data, ctx), self.spec.timeout_seconds)

        try:
            result = await retry_async(attempt, self.spec.retry, retry_on=(AllProvidersFailed, TimeoutError))
        except Exception as exc:
            scope.emit(EventType.AGENT_FAILED, error=f"{type(exc).__name__}: {exc}")
            raise
        scope.emit(EventType.AGENT_FINISHED, duration_ms=round((time.perf_counter() - started) * 1000, 3))
        return result

    # --- extension points --------------------------------------------------------------------

    async def run(self, data: I, ctx: AgentContext) -> O:
        """Default behaviour: one structured generation from the validated input."""
        return await self.generate(data, ctx, source=data)

    def check(self, output: O, source: BaseModel) -> None:
        """Semantic validation beyond the schema. Raise OutputRejected to send the model a correction."""

    # --- helpers for subclasses --------------------------------------------------------------

    async def use_tool(self, name: str, payload: BaseModel | dict, ctx: AgentContext) -> BaseModel:
        return await ctx.tools.call(self.caller, name, payload, ctx.scope)

    async def generate(self, payload: BaseModel, ctx: AgentContext, *, source: BaseModel,
                       output_model: type[BaseModel] | None = None) -> O:
        """One structured model call. `output_model` defaults to the agent's output; an agent that builds its
        output from several steps can ask for an intermediate schema instead."""
        output_model = output_model or self.spec.output_model
        body = payload.model_dump(mode="json")
        try:
            return await StructuredLLM(ctx.router).generate(
                output_model, agent_id=self.spec.id, tier=ctx.router.tier_for(self.spec.id, self.spec.tier),
                system=self.system_prompt, prompt=self._render(body, output_model.model_json_schema()),
                input_payload=body, scope=ctx.scope, max_output_tokens=self.spec.max_output_tokens,
                timeout_seconds=self.spec.llm_timeout_seconds, validation_retries=self.spec.validation_retries,
                check=lambda output: self.check(output, source),
            )
        except StructuredOutputError as exc:
            raise AgentOutputError(str(exc)) from exc

    def _render(self, body: dict, schema: dict) -> str:
        return (
            f"{self.instructions}\n\nINPUT (JSON):\n```json\n{json.dumps(body, ensure_ascii=False, indent=1)}\n```\n\n"
            f"Respond with JSON only, matching this JSON Schema:\n```json\n{json.dumps(schema)}\n```"
        )
