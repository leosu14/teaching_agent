"""TeachingSessionAgent: phrases the teacher's next turn in an interactive session, and answers learner questions
from the lesson's grounded material.

It receives the action the deterministic policy chose and returns a structured, validated turn. It cannot change the
session: its output has no field for mastery, objective status or completion, every turn must be exactly the action it
was asked for, and citations must come from the material it was given.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.teaching import (
    QUESTION_ACTIONS,
    GroundedAnswer,
    GroundingPack,
    GroundingRequest,
    PracticeItem,
    TeacherTurnOutput,
    TeachingAction,
    TeachingTurnInput,
)
from app.utils.text import reveals

AGENT_ID = "teaching_session"
LIMITATION = ("I can only answer from this lesson and its sources, and they do not cover that question. "
              "Let's stay with the lesson's topic; you can ask about it in a lesson on that subject.")


class TeachingSessionAgent(Agent[TeachingTurnInput, TeacherTurnOutput]):
    spec = AgentSpec(
        id=AGENT_ID,
        name="Interactive Teacher",
        description=("Writes the teacher's next turn in an interactive session (explanation, question, hint, feedback, "
                     "reteaching, summary) for the action the deterministic teaching policy chose, and answers learner "
                     "questions from the lesson's grounded material only."),
        input_model=TeachingTurnInput,
        output_model=TeacherTurnOutput,
        tier=ModelTier.STANDARD,
        tools=("teaching.ground", "rag.concept_map"),
        permissions=frozenset({"knowledge:read"}),
        max_output_tokens=2000,
    )
    instructions = "Write the teacher's turn for the input's action (or answer the learner's question)."

    async def run(self, data: TeachingTurnInput, ctx: AgentContext) -> TeacherTurnOutput:
        if data.stage == "turn":
            if data.action in QUESTION_ACTIONS:
                data = await self._with_probes(data, ctx)
            return await self.generate(data, ctx, source=data)
        assert data.learner_question is not None
        pack = await self.use_tool("teaching.ground", GroundingRequest(
            question=data.learner_question, subject=data.subject, candidates=data.sources), ctx)
        assert isinstance(pack, GroundingPack)
        base = {"action": TeachingAction.EXPLAIN, "concept_id": data.objective.concept_id,
                "difficulty": data.difficulty, "expected_response_type": "none"}
        if not pack.items:  # nothing in the material covers it: a structured limitation, no model call
            return TeacherTurnOutput(**base, response=LIMITATION, grounded=False,
                                     limitation="the lesson, its research and the knowledge base do not cover this "
                                                "question")
        grounded = data.model_copy(update={"sources": pack.items})
        answer = await self.generate(grounded, ctx, source=grounded, output_model=GroundedAnswer)
        assert isinstance(answer, GroundedAnswer)
        return TeacherTurnOutput(**base, response=answer.response, citations=answer.citations,
                                 grounded=answer.grounded, limitation=answer.limitation)

    async def _with_probes(self, data: TeachingTurnInput, ctx: AgentContext) -> TeachingTurnInput:
        """Questions draw on the lesson's practice items plus the knowledge base's probes for the concept."""
        if not data.topic:
            return data
        found = await self.use_tool("rag.concept_map", {"subject": data.subject, "topic": data.topic}, ctx)
        entry = next((e for e in getattr(found, "concepts", []) if e.concept.concept_id == data.objective.concept_id),
                     None)
        if entry is None or not entry.probes:
            return data
        probes = [PracticeItem(prompt=p.prompt, answer=p.answer, accepted=p.accepted) for p in entry.probes]
        return data.model_copy(update={"practice": [*data.practice, *probes]})

    def check(self, output: TeacherTurnOutput | GroundedAnswer, source: TeachingTurnInput) -> None:
        if isinstance(output, GroundedAnswer):
            allowed = {i.ref for i in source.sources} | {c for i in source.sources for c in i.citations}
            invented = sorted(set(output.citations) - allowed)
            if invented:
                raise OutputRejected(f"citations {invented} are not in the grounding material; cite only its refs")
            return
        if output.action != source.action:
            if output.action == TeachingAction.COMPLETE:
                raise OutputRejected("the teacher cannot complete the session; write the requested "
                                     f"{source.action.value} turn")
            raise OutputRejected(f"the turn must be {source.action.value}, not {output.action.value}")
        if output.concept_id != source.objective.concept_id:
            raise OutputRejected(f"the turn must be about concept {source.objective.concept_id}")
        if output.difficulty != source.difficulty:
            raise OutputRejected(f"the turn must be written at difficulty {source.difficulty}")
        if output.grounded is not None or output.limitation is not None:
            raise OutputRejected("grounded and limitation belong to answers to learner questions only")
        expected_level = source.hint_level if source.action == TeachingAction.HINT else 0
        if output.hint_level != expected_level:
            raise OutputRejected(f"hint_level must be {expected_level}")
        self._check_question(output, source)
        answers = self._answers(source)
        if source.action == TeachingAction.HINT and not source.reveal_answer and reveals(output.response, answers):
            raise OutputRejected("a hint must not reveal the expected answer")
        if source.answer_outcome == "UNCERTAIN" and reveals(output.response, answers):
            raise OutputRejected("the question stays open after an ungradable answer: do not reveal the answer")
        allowed = ({s.ref for s in source.sections} | {c for s in source.sections for c in s.citations}
                   | {i.ref for i in source.sources} | {c for i in source.sources for c in i.citations})
        invented = sorted(set(output.citations) - allowed)
        if invented:
            raise OutputRejected(f"citations {invented} do not exist in the lesson material")
        if output.misconceptions:
            if source.answer_correct is not False or source.answer_outcome not in (None, "INCORRECT"):
                raise OutputRejected("misconceptions can only be proposed for an incorrect answer")
            stray = sorted({m.concept_id for m in output.misconceptions} - {source.objective.concept_id})
            if stray:
                raise OutputRejected(f"misconceptions must be about concept {source.objective.concept_id}, not {stray}")

    @staticmethod
    def _answers(source: TeachingTurnInput) -> list[str]:
        q = source.question
        return [q.expected_answer, *q.accepted_answers] if q is not None else []

    @staticmethod
    def _check_question(output: TeacherTurnOutput, source: TeachingTurnInput) -> None:
        asks = source.action in QUESTION_ACTIONS
        if asks != (output.question is not None):
            raise OutputRejected("a question turn returns exactly one question; other turns return none")
        if output.question is None:
            if output.expected_response_type != "none":
                raise OutputRejected("expected_response_type must be 'none' for a turn that asks nothing")
            return
        q = output.question
        if output.expected_response_type != ("multiple_choice" if q.kind == "multiple_choice" else "free_text"):
            raise OutputRejected("expected_response_type must match the question kind")
        if q.kind in ("short_answer", "free_text") and reveals(q.prompt, [q.expected_answer]):
            raise OutputRejected("the question must not contain its own answer")
