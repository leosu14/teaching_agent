"""Runs the semantic grader: one validated agent call per free-text answer.

The runtime knows nothing about attempts or storage: it receives the minimal grading request and returns the agent's
validated candidate (or why there is none) with what the call used. Model calls are attributed to the task the grade
belongs to (events, cost), through the existing scope and usage ledger.
"""

from __future__ import annotations

from app.agents.assessment.agent import AGENT_ID
from app.agents.base import AgentContext
from app.agents.registry import AgentRegistry
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.providers.llm.router import ModelRouter
from app.schemas.assessment import GraderUsage, SemanticGradeCandidate, SemanticGraderResult, SemanticGradingRequest
from app.tools.manager import ToolManager

NODE_ID = "assessment"


class AgentSemanticGrader:
    """The SemanticGrader interface over the `semantic_grader` agent. `bind` attributes calls to a task (or reuses
    the caller's scope, e.g. a workflow node's, so the task's own ledger records the cost)."""

    def __init__(self, agents: AgentRegistry, tools: ToolManager, router: ModelRouter, events: EventBus) -> None:
        self._agents = agents
        self._tools = tools
        self._router = router
        self._events = events

    def bind(self, *, task_id: str | None, scope: ExecutionScope | None = None) -> BoundSemanticGrader:
        scope = scope or ExecutionScope(events=self._events, usage=UsageLedger(), task_id=task_id, node_id=NODE_ID)
        return BoundSemanticGrader(self, scope)

    def _provider_of(self, model: str | None) -> str | None:
        return next((t.provider for t in self._router.config.all_targets() if t.model == model), None)

    async def _grade(self, request: SemanticGradingRequest, scope: ExecutionScope) -> SemanticGraderResult:
        agent = self._agents.get(AGENT_ID)
        before = scope.usage.summary.by_agent.get(AGENT_ID)
        calls, tokens_in, tokens_out, cost = ((before.calls, before.usage.input_tokens, before.usage.output_tokens,
                                               before.cost_usd) if before else (0, 0, 0, 0.0))
        models_before = {m: line.calls for m, line in scope.usage.summary.by_model.items()}
        error = None
        candidate = None
        try:
            output = await agent.execute(request, AgentContext(router=self._router, tools=self._tools, scope=scope))
            assert isinstance(output, SemanticGradeCandidate)
            candidate = output
        except Exception as exc:  # invalid output after retries, every provider down, a budget error
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        after = scope.usage.summary.by_agent.get(AGENT_ID)
        used = [m for m, line in scope.usage.summary.by_model.items() if line.calls > models_before.get(m, 0)]
        model = used[-1] if used else None
        usage = GraderUsage(
            llm_calls=(after.calls - calls) if after else 0, provider=self._provider_of(model), model=model,
            input_tokens=(after.usage.input_tokens - tokens_in) if after else 0,
            output_tokens=(after.usage.output_tokens - tokens_out) if after else 0,
            estimated_cost_usd=round(after.cost_usd - cost, 8) if after else None)
        return SemanticGraderResult(candidate=candidate, error=error, usage=usage)


class BoundSemanticGrader:
    def __init__(self, owner: AgentSemanticGrader, scope: ExecutionScope) -> None:
        self._owner = owner
        self._scope = scope

    async def grade(self, request: SemanticGradingRequest) -> SemanticGraderResult:
        return await self._owner._grade(request, self._scope)
