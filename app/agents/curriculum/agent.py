"""LearningPathPlannerAgent: words a deterministic curriculum draft (objectives and their pedagogical reasoning).

The model sees only the CurriculumBrief (concepts in scope, their prerequisites and a coarse state band; no learner
or goal ids, evidence or history). It may propose objective wording, a suggested order and an explanation. It can
never set mastery, mark anything mastered or complete, add concepts or change prerequisites: the proposal schema has
no such fields, and code validates the proposal (app/curriculum/validation.py) before anything is stored.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.curriculum import CurriculumBrief, CurriculumProposal


class LearningPathPlannerAgent(Agent[CurriculumBrief, CurriculumProposal]):
    spec = AgentSpec(
        id="learning_path_planner",
        name="Learning Path Planner",
        description=("Words the objectives of a deterministic, validated curriculum and explains the learning "
                     "path; never changes its concepts, prerequisites, mastery or status."),
        input_model=CurriculumBrief,
        output_model=CurriculumProposal,
        tier=ModelTier.REASONING,
    )
    instructions = "Word the objectives of this curriculum and explain the learning path."

    def check(self, output: CurriculumProposal, source: CurriculumBrief) -> None:
        known = {c.concept_id for c in source.concepts}
        ids = [o.concept_id for o in output.objectives]
        if len(ids) != len(set(ids)):
            raise OutputRejected("propose at most one objective per concept")
        unknown = sorted(set(ids) - known)
        if unknown:
            # Unknown concepts would be rejected by validation anyway; ask the model to stay within the brief.
            raise OutputRejected(f"use only the brief's concept ids; unknown: {unknown}")
