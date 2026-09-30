"""WorkflowPlanner: maps a request's required capabilities to a workflow template and estimates cost."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.agents.registry import AgentRegistry
from app.providers.llm.router import ModelRouter
from app.runtime.workflow.engine import WorkflowDefinition
from app.schemas.lesson import LessonRequest


class NoWorkflowForRequest(LookupError):
    pass


@dataclass(frozen=True)
class ExpectedCall:
    agent_id: str
    input_tokens: int
    output_tokens: int
    calls: int = 1


@dataclass(frozen=True)
class WorkflowTemplate:
    id: str
    description: str
    provides: frozenset[str]
    build: Callable[[LessonRequest], WorkflowDefinition]
    expected_calls: tuple[ExpectedCall, ...]


class WorkflowPlanner:
    def __init__(self, templates: list[WorkflowTemplate], router: ModelRouter, agents: AgentRegistry) -> None:
        self._templates = {t.id: t for t in templates}
        self._router = router
        self._agents = agents

    def select(self, request: LessonRequest) -> WorkflowTemplate:
        needed = set(request.capabilities)
        for template in self._templates.values():
            if needed <= template.provides:
                return template
        raise NoWorkflowForRequest(f"no workflow provides capabilities {sorted(needed)}")

    def template(self, workflow_id: str) -> WorkflowTemplate:
        try:
            return self._templates[workflow_id]
        except KeyError:
            raise NoWorkflowForRequest(f"unknown workflow '{workflow_id}'") from None

    def build(self, workflow_id: str, request: LessonRequest) -> WorkflowDefinition:
        return self.template(workflow_id).build(request)

    def estimate(self, template: WorkflowTemplate) -> float:
        total = 0.0
        for call in template.expected_calls:
            tier = self._router.tier_for(call.agent_id, self._agents.get(call.agent_id).spec.tier)
            total += call.calls * self._router.estimate(tier, call.input_tokens, call.output_tokens)
        return round(total, 8)

    def templates(self) -> list[WorkflowTemplate]:
        return list(self._templates.values())
