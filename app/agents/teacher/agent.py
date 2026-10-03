"""TeacherAgent: writes structured lesson content from the plan and research."""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.lesson import LessonContent, TeacherInput


class TeacherAgent(Agent[TeacherInput, LessonContent]):
    spec = AgentSpec(
        id="teacher",
        name="Teacher",
        description="Explains concepts with examples and analogies, writes narration, exercises and questions.",
        input_model=TeacherInput,
        output_model=LessonContent,
        tier=ModelTier.REASONING,
        max_output_tokens=8000,
    )
    instructions = "Write the lesson content. If a revision is requested, fix every listed issue."

    def check(self, output: LessonContent, source: TeacherInput) -> None:
        brief = source.pedagogical_plan
        planned = {c.concept_id for c in source.plan.concepts}
        reviews = set(source.plan.review_concepts) | set(brief.review_concepts)
        # Mastered concepts are never taught again unless the plan explicitly asks for a spaced review.
        stray = [s.section_id for s in output.sections if s.concept_id not in planned | reviews
                 or s.concept_id not in set(brief.concept_ids())]
        if stray:
            raise OutputRejected(f"sections {stray} teach concepts that are not in the plan")
        expected = {o.objective_id: o.concept_id for o in brief.lesson_objectives}
        if {o.objective_id: o.concept_id for o in output.objectives} != expected:
            raise OutputRejected(f"the lesson must state exactly the plan's objectives {sorted(expected)}")
        for s in output.sections:
            if not s.objective_ids:
                raise OutputRejected(f"section {s.section_id} must reference the objectives it serves")
            if any(expected[o] != s.concept_id for o in s.objective_ids):
                raise OutputRejected(f"section {s.section_id} references objectives of another concept")
            if brief.treatment(s.concept_id).mode == "review" and s.purpose not in ("review", "assessment"):
                raise OutputRejected(f"section {s.section_id} covers a concept the plan only reviews; "
                                     "its purpose must be review")
        untaught = sorted(set(brief.target_concepts) - {s.concept_id for s in output.sections})
        if untaught:
            raise OutputRejected(f"target concepts {untaught} have no section")
        known = {c.citation_id for c in source.research.citations}
        invented = sorted({c for s in output.sections for c in s.citations} - known)
        if invented:
            raise OutputRejected(f"citations {invented} do not exist in the research bundle")
