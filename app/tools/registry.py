"""ToolRegistry: the catalogue of available tools."""

from __future__ import annotations

from app.tools.base import Tool, ToolInfo, ToolNotFound


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(f"no tool named '{name}'") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[ToolInfo]:
        return [self._tools[n].info() for n in self.names()]
