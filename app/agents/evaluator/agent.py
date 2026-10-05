"""LearnerEvaluationAgent: assesses what the learner took away from a lesson and recommends what's next."""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.evaluation import EvaluationInput, EvaluationStep
from app.schemas.events import EventType


class LearnerEvaluationAgent(Agent[EvaluationInput, EvaluationStep]):
    spec = AgentSpec(
        id="learner_evaluation",
        name="Learner Evaluation",
        description=(
            "Builds a post-lesson assessment from the lesson's objectives, taught concepts, the learner's level "
            "and mastery; grades answers; classifies concepts as mastered, partial or gaps; recommends next steps."
        ),
        input_model=EvaluationInput,
        output_model=EvaluationStep,
        tier=ModelTier.STANDARD,
    )
    instructions = "Run the requested stage of the post-lesson evaluation."

    async def run(self, data: EvaluationInput, ctx: AgentContext) -> EvaluationStep:
        if data.stage == "evaluate":
            ctx.scope.emit(EventType.EVALUATION_STARTED, answers=len(data.response.answers))
        step = await self.generate(data, ctx, source=data)
        if step.assessment is not None:
            ctx.scope.emit(EventType.ASSESSMENT_CREATED, questions=len(step.assessment.questions),
                           total_points=step.assessment.total_points)
        if step.result is not None:
            ctx.scope.emit(EventType.EVALUATION_COMPLETED, score=step.result.score, mastered=step.result.mastered,
                           partial=step.result.partial, gaps=step.result.gaps)
            ctx.scope.emit(EventType.RECOMMENDATION_CREATED, action=step.result.recommendation.action,
                           focus=step.result.recommendation.focus_concepts)
        return step

    def check(self, output: EvaluationStep, source: EvaluationInput) -> None:
        if output.stage != source.stage:
            raise OutputRejected(f"expected stage '{source.stage}'")
        taught = {s.concept_id for s in source.lesson.sections}
        if output.assessment is not None:
            stray = sorted({q.concept_id for q in output.assessment.questions} - taught)
            if stray:
                raise OutputRejected(f"questions target concepts the lesson did not teach: {stray}")
            missing = sorted(taught - {q.concept_id for q in output.assessment.questions})
            if missing:
                raise OutputRejected(f"every taught concept needs at least one question; missing {missing}")
        if output.result is not None:
            assert source.assessment is not None
            asked = {q.question_id: q for q in source.assessment.questions}
            graded = [e.question_id for e in output.result.evaluations]
            if sorted(graded) != sorted(asked):
                raise OutputRejected("grade every assessment question exactly once")
            if any(e.concept_id != asked[e.question_id].concept_id for e in output.result.evaluations):
                raise OutputRejected("evaluations must keep each question's concept")
            grades = {g.question_id: g for g in source.grades}
            disagree = sorted(e.question_id for e in output.result.evaluations
                              if e.question_id in grades and e.correct != (grades[e.question_id].outcome == "CORRECT"))
            if disagree:  # the assessment service graded the answers; the evaluation reports, never regrades
                raise OutputRejected(f"evaluations must keep the validated grades; {disagree} disagree")
            if {c.concept_id for c in output.result.concepts} != {q.concept_id for q in asked.values()}:
                raise OutputRejected("concept outcomes must cover exactly the assessed concepts")
            rec = output.result.recommendation
            unknown = sorted(set(rec.focus_concepts) - {c.concept_id for c in output.result.concepts}
                             - {c.concept_id for c in source.snapshot.concept_mastery})
            if unknown:
                raise OutputRejected(f"recommendation focuses on unknown concepts {unknown}")
