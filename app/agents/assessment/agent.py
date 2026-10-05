"""SemanticGraderAgent: proposes a candidate grade for one free-text answer against its rubric.

It sees only the question, the expected answer, the rubric, the learner's answer and the relevant lesson and research
passages. Its output is a candidate: schema-validated, then checked by the assessment validation rules (every
criterion scored once, citations only from the given passages, misconceptions only about the assessed concepts,
scores consistent with the criteria). It has no field for mastery, objective, goal or curriculum state, and the final
score and outcome are always recomputed by code.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec, OutputRejected
from app.assessment.validation import CandidateRejected, check_candidate
from app.schemas.assessment import AssessmentConfig, SemanticGradeCandidate, SemanticGradingRequest
from app.schemas.common import ModelTier

AGENT_ID = "semantic_grader"


class SemanticGraderAgent(Agent[SemanticGradingRequest, SemanticGradeCandidate]):
    spec = AgentSpec(
        id=AGENT_ID,
        name="Semantic Grader",
        description=("Grades one free-text answer against its rubric and expected meaning: criterion scores, "
                     "confidence, candidate misconceptions and feedback grounded in the lesson. Proposes; code "
                     "aggregates, classifies and decides."),
        input_model=SemanticGradingRequest,
        output_model=SemanticGradeCandidate,
        tier=ModelTier.STANDARD,
        max_output_tokens=1500,
        llm_timeout_seconds=45.0,
    )
    instructions = "Grade the learner's answer against the rubric. Respond with the candidate grade."

    def __init__(self, config: AssessmentConfig | None = None, system_prompt: str | None = None) -> None:
        super().__init__(system_prompt)
        self.config = config or AssessmentConfig()

    def check(self, output: SemanticGradeCandidate, source: SemanticGradingRequest) -> None:
        try:
            check_candidate(output, source, self.config)
        except CandidateRejected as exc:
            raise OutputRejected(str(exc)) from exc
