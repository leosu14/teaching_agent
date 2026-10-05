"""`assessment.grade`: the evaluation workflow's way to the AssessmentService.

The tool holds no grading logic. It forwards the answers to the assessment port, which the composition root binds to
the AssessmentService (the one grading pipeline), and passes the node's scope so events and model cost belong to the
evaluation task.
"""

from __future__ import annotations

from typing import Protocol

from app.observability.scope import ExecutionScope
from app.schemas.assessment import BatchAssessmentRequest, BatchAssessmentResult
from app.tools.base import Tool


class AssessmentPort(Protocol):
    async def assess_batch(self, request: BatchAssessmentRequest, scope: ExecutionScope) -> BatchAssessmentResult: ...


class AssessmentGradeTool(Tool[BatchAssessmentRequest, BatchAssessmentResult]):
    name = "assessment.grade"
    description = ("Grade learner answers through the assessment service: deterministic matching first, rubric and "
                   "semantic grading for free text; immutable grades, idempotent per attempt.")
    input_model = BatchAssessmentRequest
    output_model = BatchAssessmentResult
    permissions = frozenset({"assessment:write"})
    timeout_seconds = 600.0  # a semantic grade per free-text answer

    def __init__(self, port: AssessmentPort) -> None:
        self._port = port

    async def run(self, data: BatchAssessmentRequest, scope: ExecutionScope) -> BatchAssessmentResult:
        return await self._port.assess_batch(data, scope)
