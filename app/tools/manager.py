"""ToolManager: the only runtime path to a tool. Enforces allow-lists and permissions, validates
input and output, applies timeout and retry, and emits tool events."""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel, ValidationError

from app.observability.scope import ExecutionScope, active_scope
from app.schemas.events import EventType
from app.tools.base import ToolCaller, ToolError, ToolPermissionError, ToolTransientError
from app.tools.registry import ToolRegistry
from app.utils.retry import retry_async


class ToolManager:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def call(self, caller: ToolCaller, name: str, payload: BaseModel | dict, scope: ExecutionScope) -> BaseModel:
        tool = self._registry.get(name)
        if name not in caller.allowed_tools:
            raise ToolPermissionError(f"'{caller.caller_id}' is not allowed to use tool '{name}'")
        missing = tool.permissions - caller.permissions
        if missing:
            raise ToolPermissionError(f"'{caller.caller_id}' lacks permissions {sorted(missing)} for tool '{name}'")
        try:
            data = tool.input_model.model_validate(
                payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
            )
        except ValidationError as exc:
            raise ToolError(f"invalid input for tool '{name}': {exc}") from exc

        async def attempt(n: int) -> BaseModel:
            started = time.perf_counter()
            scope.emit(EventType.TOOL_STARTED, tool=name, caller=caller.caller_id, attempt=n)
            try:
                with active_scope(scope):  # provider calls made by the tool report into this task's scope
                    result = await asyncio.wait_for(tool.run(data, scope), tool.timeout_seconds)
            except TimeoutError as exc:
                scope.emit(EventType.TOOL_FAILED, tool=name, attempt=n, error="timeout")
                raise ToolTransientError(f"tool '{name}' timed out after {tool.timeout_seconds}s") from exc
            except ToolError as exc:
                scope.emit(EventType.TOOL_FAILED, tool=name, attempt=n, error=str(exc))
                raise
            if not isinstance(result, tool.output_model):
                scope.emit(EventType.TOOL_FAILED, tool=name, attempt=n, error="invalid output type")
                raise ToolError(f"tool '{name}' returned {type(result).__name__}, expected {tool.output_model.__name__}")
            scope.emit(EventType.TOOL_FINISHED, tool=name, attempt=n,
                       duration_ms=round((time.perf_counter() - started) * 1000, 3))
            return result

        return await retry_async(attempt, tool.retry, retry_on=(ToolTransientError,))
