"""AgentRegistry: the catalogue of agents available to workflows."""

from __future__ import annotations

from app.agents.base import Agent, AgentInfo


class UnknownAgent(KeyError):
    pass


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.spec.id in self._agents:
            raise ValueError(f"agent '{agent.spec.id}' is already registered")
        self._agents[agent.spec.id] = agent

    def get(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise UnknownAgent(agent_id) from None

    def ids(self) -> list[str]:
        return sorted(self._agents)

    def describe(self) -> list[AgentInfo]:
        return [self._agents[i].info() for i in self.ids()]
