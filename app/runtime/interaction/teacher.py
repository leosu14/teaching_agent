"""Runs the interactive teacher: one validated agent call per teacher turn.

The runtime knows nothing about sessions or storage: it receives the structured context window the service built and
returns the agent's validated output. Its model calls are attributed to the lesson task (events, cost), like every
other piece of work on that lesson.
"""

from __future__ import annotations

from app.agents.base import AgentContext
from app.agents.registry import AgentRegistry
from app.agents.teaching.agent import AGENT_ID
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.providers.llm.router import ModelRouter
from app.schemas.teaching import TeacherTurnOutput, TeachingTurnInput
from app.tools.manager import ToolManager

NODE_ID = "teaching_session"


class TeacherTurnFailed(Exception):
    """The teacher's turn could not be produced (invalid model output after retries, providers down, a tool error)."""


class TeachingRuntime:
    def __init__(self, agents: AgentRegistry, tools: ToolManager, router: ModelRouter, events: EventBus) -> None:
        self._agents = agents
        self._tools = tools
        self._router = router
        self._events = events

    async def teacher_turn(self, data: TeachingTurnInput, *, task_id: str) -> tuple[TeacherTurnOutput, UsageLedger]:
        agent = self._agents.get(AGENT_ID)
        usage = UsageLedger()
        scope = ExecutionScope(events=self._events, usage=usage, task_id=task_id, node_id=NODE_ID)
        try:
            output = await agent.execute(data, AgentContext(router=self._router, tools=self._tools, scope=scope))
        except Exception as exc:  # invalid output after retries, every provider down, a tool or budget error
            raise TeacherTurnFailed(f"{type(exc).__name__}: {exc}") from exc
        assert isinstance(output, TeacherTurnOutput)
        return output, usage
