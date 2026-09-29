"""Tool contract. Tools do the HOW (search, store, render); they contain no educational strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from app.observability.scope import ExecutionScope
from app.schemas.catalog import ToolInfo
from app.schemas.common import RetryPolicy

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)


class ToolError(Exception):
    """A tool failed and retrying will not help."""


class ToolTransientError(ToolError):
    """A tool failed in a way that may succeed on retry (network blip, rate limit)."""


class ToolPermissionError(ToolError):
    pass


class ToolNotFound(ToolError):
    pass




@dataclass(frozen=True)
class ToolCaller:
    """Who is calling: an agent (with its allow-list) or a workflow node."""

    caller_id: str
    allowed_tools: frozenset[str]
    permissions: frozenset[str] = field(default_factory=frozenset)


class Tool(ABC, Generic[I, O]):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    permissions: ClassVar[frozenset[str]] = frozenset()
    timeout_seconds: ClassVar[float] = 30.0
    retry: ClassVar[RetryPolicy] = RetryPolicy()

    @abstractmethod
    async def run(self, data: I, scope: ExecutionScope) -> O: ...

    def info(self) -> ToolInfo:
        return ToolInfo(
            name=self.name,
            description=self.description,
            permissions=sorted(self.permissions),
            timeout_seconds=self.timeout_seconds,
            max_attempts=self.retry.max_attempts,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
        )
